import asyncio
import base64
import io
from typing import Optional, Dict, Any, List, Set
from pathlib import Path
import json
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from mcp.types import Tool, TextContent, ImageContent, ServerCapabilities
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
import aiofiles


@dataclass
class BrowserSession:
    """浏览器会话管理"""
    browser: Browser
    context: BrowserContext
    page: Page
    last_used: float
    session_id: str


class HighPerformanceBrowserMCP:
    def __init__(self, max_sessions: int = 3):
        self.server = Server("high-performance-browser-server")
        self.playwright = None
        self.sessions: Dict[str, BrowserSession] = {}
        self.max_sessions = max_sessions
        self.current_session_id: Optional[str] = None
        self.page_cache: Dict[str, str] = {}  # URL -> 页面内容缓存
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 性能配置
        self.default_timeout = 3000  # 减少默认超时时间
        self.navigation_timeout = 10000
        self.screenshot_quality = 80  # 截图质量
        
        self.setup_handlers()

    def setup_handlers(self):
        """设置所有处理器"""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="create_session",
                    description="创建新的浏览器会话",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "会话ID",
                                "default": "default"
                            },
                            "headless": {
                                "type": "boolean",
                                "description": "是否无头模式",
                                "default": True
                            },
                            "viewport": {
                                "type": "object",
                                "properties": {
                                    "width": {"type": "integer", "default": 1280},
                                    "height": {"type": "integer", "default": 720}
                                }
                            }
                        }
                    }
                ),
                Tool(
                    name="fast_navigate",
                    description="快速导航到URL（优化版）",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "session_id": {"type": "string", "default": "default"},
                            "wait_until": {
                                "type": "string", 
                                "enum": ["load", "domcontentloaded", "networkidle"],
                                "default": "domcontentloaded"
                            }
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="smart_click",
                    description="智能点击（支持多种选择器）",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "session_id": {"type": "string", "default": "default"},
                            "force": {"type": "boolean", "default": False},
                            "wait_for_navigation": {"type": "boolean", "default": False}
                        },
                        "required": ["selector"]
                    }
                ),
                Tool(
                    name="batch_type",
                    description="批量输入（支持多个输入框）",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "inputs": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "selector": {"type": "string"},
                                        "text": {"type": "string"},
                                        "clear": {"type": "boolean", "default": True}
                                    }
                                }
                            },
                            "session_id": {"type": "string", "default": "default"}
                        },
                        "required": ["inputs"]
                    }
                ),
                Tool(
                    name="get_structured_content",
                    description="获取结构化页面内容",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "default": "default"},
                            "extract_links": {"type": "boolean", "default": False},
                            "extract_images": {"type": "boolean", "default": False},
                            "max_length": {"type": "integer", "default": 5000}
                        }
                    }
                ),
                Tool(
                    name="smart_screenshot",
                    description="智能截图并支持AI分析",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "default": "default"},
                            "element_selector": {
                                "type": "string",
                                "description": "特定元素截图"
                            },
                            "return_base64": {"type": "boolean", "default": True},
                            "save_file": {"type": "boolean", "default": False},
                            "filename": {"type": "string", "default": "screenshot.png"},
                            "quality": {"type": "integer", "default": 80}
                        }
                    }
                ),
                Tool(
                    name="execute_js_batch",
                    description="批量执行JavaScript",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scripts": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "session_id": {"type": "string", "default": "default"}
                        },
                        "required": ["scripts"]
                    }
                ),
                Tool(
                    name="wait_for_conditions",
                    description="等待多个条件",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "conditions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": ["selector", "url", "title", "js"]
                                        },
                                        "value": {"type": "string"}
                                    }
                                }
                            },
                            "session_id": {"type": "string", "default": "default"},
                            "timeout": {"type": "integer", "default": 5000}
                        },
                        "required": ["conditions"]
                    }
                ),
                Tool(
                    name="extract_elements_data",
                    description="批量提取元素数据",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "selectors": {
                                "type": "object",
                                "description": "键值对，键为字段名，值为选择器"
                            },
                            "session_id": {"type": "string", "default": "default"}
                        },
                        "required": ["selectors"]
                    }
                ),
                Tool(
                    name="page_performance_metrics",
                    description="获取页面性能指标",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "default": "default"}
                        }
                    }
                ),
                Tool(
                    name="close_session",
                    description="关闭指定会话",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "default": "default"}
                        }
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[TextContent | ImageContent]:
            """高性能工具调用处理"""
            try:
                # 路由到具体方法
                method_name = f"_handle_{name}"
                if hasattr(self, method_name):
                    result = await getattr(self, method_name)(arguments)
                    
                    # 处理返回结果
                    if isinstance(result, tuple) and len(result) == 2:
                        text_result, image_data = result
                        contents = [TextContent(type="text", text=text_result)]
                        if image_data:
                            contents.append(ImageContent(
                                type="image",
                                data=image_data,
                                mimeType="image/png"
                            ))
                        return contents
                    else:
                        return [TextContent(type="text", text=str(result))]
                else:
                    return [TextContent(type="text", text=f"未知工具: {name}")]

            except Exception as e:
                return [TextContent(type="text", text=f"执行出错: {str(e)}")]

    async def _get_or_create_session(self, session_id: str = "default") -> Optional[BrowserSession]:
        """获取或创建会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_used = time.time()
            return session
        
        # 清理过期会话
        await self._cleanup_old_sessions()
        
        # 如果达到最大会话数，关闭最旧的
        if len(self.sessions) >= self.max_sessions:
            oldest_id = min(self.sessions.keys(), 
                           key=lambda x: self.sessions[x].last_used)
            await self._close_session(oldest_id)
        
        return None

    async def _cleanup_old_sessions(self, max_idle_time: int = 300):
        """清理空闲会话"""
        current_time = time.time()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_used > max_idle_time:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            await self._close_session(session_id)

    async def _close_session(self, session_id: str):
        """关闭指定会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            try:
                await session.page.close()
                await session.context.close()
            except:
                pass
            del self.sessions[session_id]

    # 具体的工具实现方法
    async def _handle_create_session(self, args: dict) -> str:
        """创建新会话"""
        session_id = args.get("session_id", "default")
        headless = args.get("headless", True)
        viewport = args.get("viewport", {"width": 1280, "height": 720})
        
        if not self.playwright:
            self.playwright = await async_playwright().start()
        
        # 关闭现有会话
        if session_id in self.sessions:
            await self._close_session(session_id)
        
        try:
            browser = await self.playwright.chromium.launch(
                headless=headless,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-first-run',
                    '--disable-default-apps',
                    '--disable-background-timer-throttling'
                ]
            )
            
            context = await browser.new_context(
                viewport=viewport,
                ignore_https_errors=True
            )
            
            page = await context.new_page()
            
            # 优化设置
            await page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            session = BrowserSession(
                browser=browser,
                context=context,
                page=page,
                last_used=time.time(),
                session_id=session_id
            )
            
            self.sessions[session_id] = session
            self.current_session_id = session_id
            
            return f"会话 {session_id} 创建成功"
            
        except Exception as e:
            return f"创建会话失败: {str(e)}"

    async def _handle_fast_navigate(self, args: dict) -> str:
        """快速导航"""
        url = args.get("url")
        session_id = args.get("session_id", "default")
        wait_until = args.get("wait_until", "domcontentloaded")
        
        session = await self._get_or_create_session(session_id)
        if not session:
            # 自动创建会话
            await self._handle_create_session({"session_id": session_id})
            session = self.sessions[session_id]
        
        try:
            # 检查缓存
            if url in self.page_cache:
                cache_time = time.time() - 60  # 1分钟缓存
                if self.page_cache.get(f"{url}_time", 0) > cache_time:
                    return f"从缓存导航到: {url}"
            
            await session.page.goto(
                url, 
                wait_until=wait_until,
                timeout=self.navigation_timeout
            )
            
            # 更新缓存
            self.page_cache[url] = url
            self.page_cache[f"{url}_time"] = time.time()
            
            return f"快速导航到: {url}"
            
        except Exception as e:
            return f"导航失败: {str(e)}"

    async def _handle_smart_click(self, args: dict) -> str:
        """智能点击"""
        selector = args.get("selector")
        session_id = args.get("session_id", "default")
        force = args.get("force", False)
        wait_for_navigation = args.get("wait_for_navigation", False)
        
        session = await self._get_or_create_session(session_id)
        if not session:
            return "会话不存在，请先创建会话"
        
        try:
            # 等待元素可见
            await session.page.wait_for_selector(
                selector, 
                state="visible",
                timeout=self.default_timeout
            )
            
            if wait_for_navigation:
                async with session.page.expect_navigation(timeout=self.navigation_timeout):
                    await session.page.click(selector, force=force, timeout=self.default_timeout)
            else:
                await session.page.click(selector, force=force, timeout=self.default_timeout)
            
            return f"智能点击成功: {selector}"
            
        except Exception as e:
            return f"点击失败: {str(e)}"

    async def _handle_batch_type(self, args: dict) -> str:
        """批量输入"""
        inputs = args.get("inputs", [])
        session_id = args.get("session_id", "default")
        
        session = await self._get_or_create_session(session_id)
        if not session:
            return "会话不存在，请先创建会话"
        
        results = []
        for input_item in inputs:
            selector = input_item.get("selector")
            text = input_item.get("text")
            clear = input_item.get("clear", True)
            
            try:
                if clear:
                    await session.page.fill(selector, text)
                else:
                    await session.page.type(selector, text)
                results.append(f"✓ {selector}")
            except Exception as e:
                results.append(f"✗ {selector}: {str(e)}")
        
        return f"批量输入完成:\n" + "\n".join(results)

    async def _handle_get_structured_content(self, args: dict) -> str:
        """获取结构化内容"""
        session_id = args.get("session_id", "default")
        extract_links = args.get("extract_links", False)
        extract_images = args.get("extract_images", False)
        max_length = args.get("max_length", 5000)
        
        session = await self._get_or_create_session(session_id)
        if not session:
            return "会话不存在，请先创建会话"
        
        try:
            content = {
                "url": session.page.url,
                "title": await session.page.title(),
                "text": ""
            }
            
            # 获取文本内容
            text_content = await session.page.text_content("body")
            content["text"] = text_content[:max_length] if text_content else ""
            
            # 提取链接
            if extract_links:
                links = await session.page.evaluate("""
                    Array.from(document.querySelectorAll('a[href]')).map(a => ({
                        text: a.textContent.trim(),
                        href: a.href
                    })).slice(0, 20)
                """)
                content["links"] = links
            
            # 提取图片
            if extract_images:
                images = await session.page.evaluate("""
                    Array.from(document.querySelectorAll('img[src]')).map(img => ({
                        alt: img.alt,
                        src: img.src
                    })).slice(0, 10)
                """)
                content["images"] = images
            
            return json.dumps(content, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return f"获取内容失败: {str(e)}"

    async def _handle_smart_screenshot(self, args: dict) -> tuple:
        """智能截图"""
        session_id = args.get("session_id", "default")
        element_selector = args.get("element_selector")
        return_base64 = args.get("return_base64", True)
        save_file = args.get("save_file", False)
        filename = args.get("filename", "screenshot.png")
        quality = args.get("quality", 80)
        
        session = await self._get_or_create_session(session_id)
        if not session:
            return "会话不存在，请先创建会话", None
        
        try:
            screenshot_options = {
                "type": "png",
                "quality": quality
            }
            
            if element_selector:
                element = await session.page.query_selector(element_selector)
                if element:
                    screenshot_bytes = await element.screenshot(**screenshot_options)
                else:
                    return f"未找到元素: {element_selector}", None
            else:
                screenshot_bytes = await session.page.screenshot(**screenshot_options)
            
            result_text = "截图完成"
            image_data = None
            
            # 保存文件
            if save_file:
                async with aiofiles.open(filename, 'wb') as f:
                    await f.write(screenshot_bytes)
                result_text += f"，已保存到 {filename}"
            
            # 返回base64数据用于AI分析
            if return_base64:
                image_data = base64.b64encode(screenshot_bytes).decode('utf-8')
                result_text += "，已生成base64数据供AI分析"
            
            return result_text, image_data
            
        except Exception as e:
            return f"截图失败: {str(e)}", None

    async def _handle_execute_js_batch(self, args: dict) -> str:
        """批量执行JavaScript"""
        scripts = args.get("scripts", [])
        session_id = args.get("session_id", "default")
        
        session = await self._get_or_create_session(session_id)
        if not session:
            return "会话不存在，请先创建会话"
        
        results = []
        for i, script in enumerate(scripts):
            try:
                result = await session.page.evaluate(script)
                results.append(f"脚本 {i+1}: {result}")
            except Exception as e:
                results.append(f"脚本 {i+1} 失败: {str(e)}")
        
        return "\n".join(results)

    async def _handle_wait_for_conditions(self, args: dict) -> str:
        """等待多个条件"""
        conditions = args.get("conditions", [])
        session_id = args.get("session_id", "default")
        timeout = args.get("timeout", 5000)
        
        session = await self._get_or_create_session(session_id)
        if not session:
            return "会话不存在，请先创建会话"
        
        results = []
        for condition in conditions:
            condition_type = condition.get("type")
            value = condition.get("value")
            
            try:
                if condition_type == "selector":
                    await session.page.wait_for_selector(value, timeout=timeout)
                elif condition_type == "url":
                    await session.page.wait_for_url(value, timeout=timeout)
                elif condition_type == "title":
                    await session.page.wait_for_function(
                        f"document.title.includes('{value}')", timeout=timeout
                    )
                elif condition_type == "js":
                    await session.page.wait_for_function(value, timeout=timeout)
                
                results.append(f"✓ {condition_type}: {value}")
            except Exception as e:
                results.append(f"✗ {condition_type}: {value} - {str(e)}")
        
        return "条件等待结果:\n" + "\n".join(results)

    async def _handle_extract_elements_data(self, args: dict) -> str:
        """批量提取元素数据"""
        selectors = args.get("selectors", {})
        session_id = args.get("session_id", "default")
        
        session = await self._get_or_create_session(session_id)
        if not session:
            return "会话不存在，请先创建会话"
        
        data = {}
        for field_name, selector in selectors.items():
            try:
                elements = await session.page.query_selector_all(selector)
                if len(elements) == 1:
                    data[field_name] = await elements[0].text_content()
                else:
                    data[field_name] = [await el.text_content() for el in elements]
            except Exception as e:
                data[field_name] = f"提取失败: {str(e)}"
        
        return json.dumps(data, ensure_ascii=False, indent=2)

    async def _handle_page_performance_metrics(self, args: dict) -> str:
        """获取页面性能指标"""
        session_id = args.get("session_id", "default")
        
        session = await self._get_or_create_session(session_id)
        if not session:
            return "会话不存在，请先创建会话"
        
        try:
            metrics = await session.page.evaluate("""
                () => {
                    const perf = performance;
                    const navigation = perf.getEntriesByType('navigation')[0];
                    return {
                        loadTime: navigation.loadEventEnd - navigation.loadEventStart,
                        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                        networkLatency: navigation.responseStart - navigation.requestStart,
                        pageSize: document.documentElement.outerHTML.length,
                        resourceCount: perf.getEntriesByType('resource').length
                    };
                }
            """)
            
            return json.dumps(metrics, indent=2)
            
        except Exception as e:
            return f"获取性能指标失败: {str(e)}"

    async def _handle_close_session(self, args: dict) -> str:
        """关闭会话"""
        session_id = args.get("session_id", "default")
        
        if session_id in self.sessions:
            await self._close_session(session_id)
            return f"会话 {session_id} 已关闭"
        else:
            return f"会话 {session_id} 不存在"

    async def cleanup(self):
        """清理所有资源"""
        for session_id in list(self.sessions.keys()):
            await self._close_session(session_id)
        
        if self.playwright:
            await self.playwright.stop()
        
        self.executor.shutdown(wait=True)

    async def run(self):
        """运行高性能MCP服务器"""
        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="high-performance-browser-server",
                        server_version="1.0.0",
                        capabilities=ServerCapabilities(
                            tools={}
                        )
                    )
                )
        finally:
            await self.cleanup()


async def main():
    server = HighPerformanceBrowserMCP(max_sessions=3)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())