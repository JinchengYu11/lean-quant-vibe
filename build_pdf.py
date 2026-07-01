import os
import sys

# Disables proxy
os.environ['NO_PROXY'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

from playwright.sync_api import sync_playwright

def main():
    print("Launching Playwright to build PDF...")
    try:
        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Load local HTML report
            html_path = os.path.abspath("report.html")
            print(f"Loading HTML file: {html_path}")
            page.goto(f"file://{html_path}")
            
            # Wait for any image/charts to load
            page.wait_for_timeout(2000)
            
            output_paths = [
                os.path.abspath("PROJECT_REPORT_CSI1000.pdf"),
                os.path.abspath(os.path.join("reports", "PROJECT_REPORT_CSI1000.pdf"))
            ]
            
            # Check if we have an environment variable for brain folder (to keep it compatible with AI systems)
            brain_dir = os.environ.get("GEMINI_BRAIN_DIR")
            if brain_dir:
                output_paths.append(os.path.abspath(os.path.join(brain_dir, "PROJECT_REPORT_CSI1000.pdf")))
            
            success = False
            for path in output_paths:
                try:
                    # Make sure parent directory exists
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    print(f"Attempting to generate PDF at: {path}")
                    page.pdf(
                        path=path,
                        format="A4",
                        print_background=True,
                        prefer_css_page_size=True
                    )
                    print(f"PDF built successfully at: {path}")
                    success = True
                except Exception as e:
                    print(f"Failed at {path}: {e}")
            if not success:
                raise Exception("All output paths failed due to file locks or permissions.")
            
            browser.close()
    except Exception as e:
        print(f"Failed to generate PDF: {e}")

if __name__ == "__main__":
    main()
