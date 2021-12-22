import pytest
from pathlib import Path

from typing import Union

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys

PATH: str = "http://localhost:8080"


class DriverWrapper:
    def __init__(self, config):
        self.headless = not config.getoption("head")
        self.config = config

        if config.getoption("driver") == "firefox":
            options = webdriver.FirefoxOptions()
            options.headless = self.headless

            self.driver = webdriver.Firefox(options=options)
        elif config.getoption("driver") == "chrome":
            options = webdriver.ChromeOptions()
            options.headless = self.headless

            self.driver = webdriver.Chrome(options=options)

        self.driver.set_window_size(1280, 720)

    def wait_complete(self, timeout=10):
        WebDriverWait(self.driver, timeout).until(
            lambda driver: driver.execute_script(
                "return window.display !== undefined && window.display.runningSign.style.display == 'none'"
            )
        )

    def main_svg(self):
        return self.driver.execute_script("return window.mainSseq.chart.svg")

    def check_svg(self, path: str):
        self.driver.execute_script("window.mainSseq.sort()");
        svg = self.main_svg().get_attribute("outerHTML")
        check_file(
            path,
            svg,
            self.config,
        )

    def check_pages(self, suffix: str, max_page: int):
        self.main_svg().click()
        self.wait_complete()

        for page in range(2, max_page + 1):
            self.check_svg(f"{suffix}_e{page}.svg")
            self.send_keys(Keys.RIGHT)

        for _ in range(2, max_page + 1):
            self.send_keys(Keys.LEFT)

    def go(self, path: str):
        self.driver.get(PATH + path)

    def click_class(self, x: int, y: int):
        self.main_svg().find_element_by_css_selector(
            f"g [data-x='{x}'][data-y='{y}'] > circle"
        ).click()

    def send_keys(self, key: str):
        self.driver.find_element_by_tag_name("body").send_keys(key)

    def select_panel(self, name: str):
        head = self.driver.execute_script("return window.display.currentPanel.head")

        found = False
        for child in head.find_elements_by_css_selector("a"):
            if child.text == name:
                child.click()
                found = True
                break

        if not found:
            raise ValueError(f"Panel {name} not found")

    def panel(self):
        return self.driver.execute_script("return window.display.currentPanel.inner")

    def sidebar(self):
        return self.driver.execute_script("return window.display.sidebar")

    def click_button(self, text: str):
        """Click the button in the side bar with the given text"""
        for elt in self.sidebar().find_elements_by_tag_name("button"):
            if elt.text == text:
                elt.click()
                return

        raise ValueError(f"Button {text} not found")

    def reply(self, text: Union[str, bool]):
        alert = self.driver.switch_to.alert
        if isinstance(text, str):
            alert.send_keys(text)
            alert.accept()

        elif text:
            alert.accept()
        else:
            alert.reject()


def pytest_addoption(parser):
    parser.addoption(
        "--head", action="store_true", help="Don't run the browser in headless mode"
    )
    parser.addoption("--update", action="store_true", help="Update benchmarks")
    parser.addoption(
        "--driver",
        default="firefox",
        action="store",
        help="Driver to use (firefox or chrome)",
    )


@pytest.fixture(scope="session")
def driver(pytestconfig):
    driver = DriverWrapper(pytestconfig)
    yield driver
    if driver.headless:
        driver.driver.quit()


def find_data(svg: str) -> str:
    """
    Given an svg, find the row that contains the actual chart, as opposed to
    the display data (e.g. transforms). This part should be stable across
    browsers
    """
    svg = svg.replace(' style=""', "")
    for line in svg.split("\n"):
        if line.startswith("</style>"):
            return line
    raise ValueError("Content line not found")


def check_file(filename: str, value: str, config):
    filename = Path(__file__).parent / "benchmarks" / filename

    if config.getoption("update"):
        with open(filename, "w") as f:
            f.write(value)

        return

    try:
        with open(filename) as f:
            bench = f.read()
    except OSError:
        bench = ""

    if find_data(bench) != find_data(value):
        new_path = filename.parent / (filename.stem + "-new.svg")
        with open(new_path, "w") as f:
            f.write(value)

        raise ValueError(
            f"{filename.name} changed. New version saved at {filename.stem}-new.svg"
        )
