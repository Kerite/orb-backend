from playwright.sync_api import sync_playwright

def extract_chatgpt_share_from_link(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        # 等待对话内容加载
        page.wait_for_selector('[data-message-author-role]', timeout=10000)
        elements = page.query_selector_all('[data-message-author-role]')

        result = []
        for el in elements:
            role = el.get_attribute("data-message-author-role")
            content = el.inner_text().strip()
            result.append({
                "role": role,
                "content": content
            })

        browser.close()
        return result
