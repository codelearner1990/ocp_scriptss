from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

def login_to_openshift(region, server_url, username, password):
    print(f"Logging into OpenShift region: {region}")
    
    # Start the browser session
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(options=options)
    
    try:
        # Access the login page via `oc login --web`
        driver.get(f"{server_url}/oauth/token/request")
        
        # Automate login flow
        driver.find_element(By.ID, "username").send_keys(username)
        driver.find_element(By.ID, "password").send_keys(password)
        driver.find_element(By.NAME, "login").click()
        
        # Extract token from the page
        time.sleep(3)  # Wait for the page to load
        token_element = driver.find_element(By.TAG_NAME, "pre")
        token = token_element.text.strip()
        
        print(f"Token retrieved: {token}")
        return token
    except Exception as e:
        print(f"Failed to login to OpenShift: {e}")
        return None
    finally:
        driver.quit()
