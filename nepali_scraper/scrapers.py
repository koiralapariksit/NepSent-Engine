"""
Domain-specific web scrapers for Nepali text collection.
Implements platform-specific crawling strategies for news, e-commerce, and social media.
"""

import requests
from bs4 import BeautifulSoup
import logging
import time
import random
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class BaseScraper:
    """
    Abstract base class for all domain-specific scrapers.
    Implements common HTTP handling, error recovery, and data persistence.
    """
    
    def __init__(self, config: Dict, headers: Dict = None, rate_limit: float = 2.0):
        """
        Initialize base scraper with shared configuration.
        
        Args:
            config: Platform-specific configuration dict
            headers: HTTP headers for requests
            rate_limit: Base delay (seconds) between requests
        """
        self.config = config
        self.headers = headers or {}
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.collected_data = []
        self.errors = []
    
    def _make_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """
        Make HTTP request with exponential backoff retry logic.
        
        Args:
            url: Target URL
            max_retries: Maximum retry attempts
            
        Returns:
            Response object or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                # Apply rate limiting with jitter
                time.sleep(self.rate_limit + random.uniform(0, 2))
                
                response = self.session.get(
                    url,
                    timeout=10,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                logger.debug(f"✓ Request succeeded: {url}")
                return response
            
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}. "
                    f"Retrying in {wait_time}s..."
                )
                
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    self.errors.append({"url": url, "error": str(e)})
                    logger.error(f"✗ Final attempt failed: {url}")
                    return None
    
    def _extract_text_paragraphs(self, soup: BeautifulSoup, selectors: List[str]) -> List[str]:
        """
        Extract text content using multiple CSS selectors in priority order.
        
        Args:
            soup: BeautifulSoup parsed HTML
            selectors: List of CSS selectors to try
            
        Returns:
            List of text paragraphs
        """
        paragraphs = []
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text:
                    paragraphs.append(text)
        
        return paragraphs
    
    def save_checkpoint(self, filename: Path):
        """
        Save collected data to JSONL file (checkpoint for resumability).
        
        Args:
            filename: Output file path
        """
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'a', encoding='utf-8') as f:
            for record in self.collected_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(self.collected_data)} records to {filename}")
    
    def get_summary(self) -> Dict:
        """Return summary statistics."""
        return {
            "collected": len(self.collected_data),
            "errors": len(self.errors),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# NEWS & POLITICS SCRAPERS
# ============================================================================

class RatopatiScraper(BaseScraper):
    """Scraper for Ratopati news portal (ratopati.com)."""
    
    def scrape_articles(self, max_pages: int = 15) -> List[Dict]:
        """
        Scrape news articles from Ratopati.com
        
        Strategy: Paginated listing with URL parameter page= pagination
        """
        logger.info(f"Starting Ratopati scrape (max {max_pages} pages)...")
        
        for page in range(1, max_pages + 1):
            url = f"https://ratopati.com/news?page={page}"
            
            response = self._make_request(url)
            if not response:
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Ratopati article listing structure
            articles = soup.select('div.article-listing article')
            
            if not articles:
                logger.warning(f"No articles found on page {page}")
                break
            
            for article in articles:
                try:
                    # Extract headline and link
                    headline_elem = article.select_one('h2 a, h3 a')
                    if not headline_elem:
                        continue
                    
                    link = headline_elem.get('href', '')
                    headline = headline_elem.get_text(strip=True)
                    
                    if not link.startswith('http'):
                        link = "https://ratopati.com" + link
                    
                    # Fetch full article content
                    article_response = self._make_request(link)
                    if not article_response:
                        continue
                    
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                    
                    # Extract article body
                    body_elem = article_soup.select_one('div.article-content, article.article-body')
                    if body_elem:
                        content = body_elem.get_text(strip=True)
                    else:
                        content = ""
                    
                    if content:
                        self.collected_data.append({
                            "domain": "news_politics",
                            "platform": "ratopati",
                            "url": link,
                            "headline": headline,
                            "content": content,
                            "scraped_at": datetime.now().isoformat()
                        })
                        logger.debug(f"✓ Article collected: {headline[:50]}...")
                
                except Exception as e:
                    logger.error(f"Error parsing article: {str(e)}")
                    self.errors.append({"url": link, "error": str(e)})
            
            logger.info(f"Page {page}: {len(articles)} articles processed")
        
        return self.collected_data


class SetopatiScraper(BaseScraper):
    """Scraper for Setopati news portal (setopati.com)."""
    
    def scrape_articles(self, max_pages: int = 15) -> List[Dict]:
        """Scrape news articles from Setopati.com"""
        logger.info(f"Starting Setopati scrape (max {max_pages} pages)...")
        
        for page in range(1, max_pages + 1):
            url = f"https://setopati.com/news?page={page}"
            
            response = self._make_request(url)
            if not response:
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.select('article.article-box, div.article-item')
            
            if not articles:
                logger.warning(f"No articles found on page {page}")
                break
            
            for article in articles:
                try:
                    # Setopati structure
                    link_elem = article.select_one('a')
                    headline_elem = article.select_one('h2, h3, .article-title')
                    
                    if not link_elem or not headline_elem:
                        continue
                    
                    link = link_elem.get('href', '')
                    headline = headline_elem.get_text(strip=True)
                    
                    if not link.startswith('http'):
                        link = "https://setopati.com" + link
                    
                    # Fetch article content
                    article_response = self._make_request(link)
                    if not article_response:
                        continue
                    
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                    body_elem = article_soup.select_one('div.article-body, article')
                    
                    content = body_elem.get_text(strip=True) if body_elem else ""
                    
                    if content:
                        self.collected_data.append({
                            "domain": "news_politics",
                            "platform": "setopati",
                            "url": link,
                            "headline": headline,
                            "content": content,
                            "scraped_at": datetime.now().isoformat()
                        })
                        logger.debug(f"✓ Article collected: {headline[:50]}...")
                
                except Exception as e:
                    logger.error(f"Error parsing article: {str(e)}")
            
            logger.info(f"Page {page}: {len(articles)} articles processed")
        
        return self.collected_data


class OnlineKhabarScraper(BaseScraper):
    """Scraper for Onlinekhabar news portal (onlinekhabar.com)."""
    
    def scrape_articles(self, max_pages: int = 15) -> List[Dict]:
        """Scrape news articles from Onlinekhabar.com"""
        logger.info(f"Starting Onlinekhabar scrape (max {max_pages} pages)...")
        
        for page in range(1, max_pages + 1):
            url = f"https://www.onlinekhabar.com/news?page={page}"
            
            response = self._make_request(url)
            if not response:
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.select('article, div.article-list-item')
            
            if not articles:
                logger.warning(f"No articles found on page {page}")
                break
            
            for article in articles:
                try:
                    link_elem = article.select_one('a')
                    headline_elem = article.select_one('h2, h3, .title')
                    
                    if not link_elem or not headline_elem:
                        continue
                    
                    link = link_elem.get('href', '')
                    headline = headline_elem.get_text(strip=True)
                    
                    if not link.startswith('http'):
                        link = "https://www.onlinekhabar.com" + link
                    
                    # Fetch article
                    article_response = self._make_request(link)
                    if not article_response:
                        continue
                    
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                    body_elem = article_soup.select_one('div.article-content, article, div.article-body')
                    
                    content = body_elem.get_text(strip=True) if body_elem else ""
                    
                    if content:
                        self.collected_data.append({
                            "domain": "news_politics",
                            "platform": "onlinekhabar",
                            "url": link,
                            "headline": headline,
                            "content": content,
                            "scraped_at": datetime.now().isoformat()
                        })
                        logger.debug(f"✓ Article collected: {headline[:50]}...")
                
                except Exception as e:
                    logger.error(f"Error parsing article: {str(e)}")
            
            logger.info(f"Page {page}: {len(articles)} articles processed")
        
        return self.collected_data


# ============================================================================
# E-COMMERCE SCRAPERS
# ============================================================================

class DarazScraper(BaseScraper):
    """Scraper for Daraz.com.np E-commerce product reviews."""
    
    def scrape_product_reviews(self, max_products: int = 100) -> List[Dict]:
        """
        Scrape product reviews from Daraz Nepal.
        
        Strategy: Iterate through popular product categories and extract reviews
        """
        logger.info(f"Starting Daraz scrape (max {max_products} products)...")
        
        # Sample categories on Daraz Nepal
        categories = [
            "electronics",
            "fashion",
            "beauty",
            "home-garden",
            "sports-outdoors"
        ]
        
        product_count = 0
        
        for category in categories:
            if product_count >= max_products:
                break
            
            # List products in category
            url = f"https://www.daraz.com.np/{category}/"
            
            response = self._make_request(url)
            if not response:
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find product links (Daraz uses <a> with data attributes)
            product_links = soup.select('a[href*="/p/"]')
            
            for link_elem in product_links:
                if product_count >= max_products:
                    break
                
                try:
                    product_url = link_elem.get('href', '')
                    if not product_url.startswith('http'):
                        product_url = "https://www.daraz.com.np" + product_url
                    
                    # Fetch product page
                    product_response = self._make_request(product_url)
                    if not product_response:
                        continue
                    
                    product_soup = BeautifulSoup(product_response.content, 'html.parser')
                    
                    # Extract product name
                    product_name = ""
                    name_elem = product_soup.select_one('h1')
                    if name_elem:
                        product_name = name_elem.get_text(strip=True)
                    
                    # Extract reviews
                    review_elements = product_soup.select('div.review-box, div.review-item, [data-review]')
                    
                    for review_elem in review_elements:
                        review_text = review_elem.get_text(strip=True)
                        
                        if len(review_text) > 15:
                            self.collected_data.append({
                                "domain": "ecommerce_reviews",
                                "platform": "daraz",
                                "product_name": product_name,
                                "url": product_url,
                                "review_text": review_text,
                                "scraped_at": datetime.now().isoformat()
                            })
                            logger.debug(f"✓ Review collected: {review_text[:50]}...")
                    
                    product_count += 1
                    logger.debug(f"Product {product_count}: {len(review_elements)} reviews")
                
                except Exception as e:
                    logger.error(f"Error parsing product: {str(e)}")
                    self.errors.append({"url": product_url, "error": str(e)})
        
        return self.collected_data


# ============================================================================
# SOCIAL MEDIA SCRAPERS
# ============================================================================

class TwitterScraper(BaseScraper):
    """
    Scraper for Twitter/X Nepali-language tweets.
    
    Note: Requires Twitter API Bearer Token for official API access.
    This implementation is a framework placeholder for API integration.
    """
    
    def __init__(self, config: Dict, bearer_token: str = None, headers: Dict = None):
        """
        Initialize Twitter scraper with API credentials.
        
        Args:
            config: Platform configuration
            bearer_token: Twitter API v2 Bearer token
            headers: Custom headers
        """
        super().__init__(config, headers)
        self.bearer_token = bearer_token
        self.api_base = "https://api.twitter.com/2"
    
    def _get_twitter_headers(self) -> Dict:
        """Build Twitter API authentication headers."""
        headers = dict(self.headers)
        if self.bearer_token:
            headers['Authorization'] = f'Bearer {self.bearer_token}'
        return headers
    
    def scrape_nepali_tweets(self, query: str = "lang:ne", max_results: int = 100) -> List[Dict]:
        """
        Search for Nepali-language tweets.
        
        Args:
            query: Search query (default: lang:ne for Nepali)
            max_results: Max tweets to collect
            
        Returns:
            List of tweet records
        """
        logger.info(f"Starting Twitter scrape (query: '{query}', max {max_results})...")
        
        if not self.bearer_token:
            logger.error(
                "⚠ Twitter API Bearer token not configured. "
                "Set TWITTER_BEARER_TOKEN environment variable."
            )
            return []
        
        try:
            # Twitter API v2 search endpoint
            url = f"{self.api_base}/tweets/search/recent"
            
            params = {
                "query": query,
                "max_results": min(100, max_results),
                "tweet.fields": "created_at,author_id,public_metrics",
                "expansions": "author_id",
                "user.fields": "username,created_at"
            }
            
            headers = self._get_twitter_headers()
            
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            tweets = data.get('data', [])
            
            for tweet in tweets:
                self.collected_data.append({
                    "domain": "social_media",
                    "platform": "twitter",
                    "tweet_id": tweet.get('id'),
                    "text": tweet.get('text'),
                    "created_at": tweet.get('created_at'),
                    "public_metrics": tweet.get('public_metrics', {}),
                    "scraped_at": datetime.now().isoformat()
                })
                logger.debug(f"✓ Tweet collected: {tweet.get('text', '')[:50]}...")
            
            logger.info(f"Collected {len(tweets)} tweets")
        
        except Exception as e:
            logger.error(f"Twitter API error: {str(e)}")
            self.errors.append({"query": query, "error": str(e)})
        
        return self.collected_data
