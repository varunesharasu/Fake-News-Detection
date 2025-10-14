import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler

class NewsScraper:
    def __init__(self):
        self.base_url = "https://timesofindia.indiatimes.com/"
        self.news_data_file = "news_data.json"
        self.load_existing_news()

    def load_existing_news(self):
        if os.path.exists(self.news_data_file):
            with open(self.news_data_file, 'r', encoding='utf-8') as f:
                self.news_data = json.load(f)
        else:
            self.news_data = {}

    def save_news_data(self):
        with open(self.news_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.news_data, f, ensure_ascii=False, indent=2)

    def scrape_news(self):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(self.base_url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find news articles - adjust selectors based on TOI's structure
            news_articles = []

            # Look for various news containers
            selectors = [
                'div.news-card',
                'div.article',
                'div.top-story',
                'div.list-item',
                'a[href*="/articleshow/"]'
            ]

            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    title = self.extract_title(element)
                    if title and len(title.strip()) > 10:  # Filter out very short titles
                        news_articles.append({
                            'title': title.strip(),
                            'url': self.extract_url(element),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'Times of India'
                        })

            # Remove duplicates and add new articles
            new_articles = 0
            for article in news_articles:
                article_key = article['title'].lower().replace(' ', '')
                if article_key not in self.news_data:
                    self.news_data[article_key] = article
                    new_articles += 1

            if new_articles > 0:
                self.save_news_data()
                print(f"Scraped {new_articles} new articles. Total articles: {len(self.news_data)}")
            else:
                print("No new articles found.")

        except Exception as e:
            print(f"Error scraping news: {e}")

    def extract_title(self, element):
        # Try different title selectors
        title_selectors = ['h2', 'h3', 'h4', '.title', '.headline', 'a']
        for selector in title_selectors:
            title_elem = element.select_one(selector)
            if title_elem:
                return title_elem.get_text().strip()
        return element.get_text().strip() if element.get_text() else None

    def extract_url(self, element):
        link = element.find('a') or element
        if link and link.get('href'):
            href = link['href']
            if href.startswith('/'):
                return self.base_url.rstrip('/') + href
            elif href.startswith('http'):
                return href
        return self.base_url

    def check_news_exists(self, user_input):
        # Simple text matching - check if user input contains or is contained in any stored news
        user_text = user_input.lower().strip()

        for article_key, article in self.news_data.items():
            article_title = article['title'].lower()
            # Check for substantial overlap
            if (user_text in article_title or
                article_title in user_text or
                self.calculate_similarity(user_text, article_title) > 0.7):
                return True, article
        return False, None

    def calculate_similarity(self, text1, text2):
        # Simple Jaccard similarity for basic matching
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0

# Global scraper instance
scraper = NewsScraper()

def start_scraping_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=scraper.scrape_news, trigger="interval", minutes=30)
    scheduler.start()

    # Initial scrape
    scraper.scrape_news()

    print("News scraping scheduler started. Will scrape every 30 minutes.")

if __name__ == "__main__":
    start_scraping_scheduler()
    # Keep the script running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping scraper...")