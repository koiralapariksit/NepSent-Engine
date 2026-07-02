"""
Main orchestration script for Nepali sentiment analysis web scraping.

Coordinates multi-domain data collection with checkpointing, error recovery,
and progress tracking.
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from nepali_scraper.config import (
    DATA_DIR, LOGS_DIR, LOG_FORMAT, LOG_LEVEL,
    DOMAIN_TARGETS, OUTPUT_FILES, PLATFORM_CONFIG
)
from nepali_scraper.processors import NepaliTextProcessor, is_valid_nepali_text
from nepali_scraper.scrapers import (
    RatopatiScraper, SetopatiScraper, OnlineKhabarScraper,
    DarazScraper, TwitterScraper
)
from nepali_scraper.utils import (
    setup_logging, generate_domain_balance_report,
    generate_scraping_report, print_report, validate_jsonl_file,
    verify_dependencies, load_environment_variables
)
import logging

# ============================================================================
# GLOBAL LOGGER
# ============================================================================

logger = setup_logging(LOGS_DIR / "scraper.log", LOG_LEVEL)
logger = logging.getLogger(__name__)


# ============================================================================
# ORCHESTRATOR CLASS
# ============================================================================

class NepaliCorpusOrchestrator:
    """
    Master orchestrator for multi-domain Nepali text collection.
    
    Responsibilities:
    - Coordinate scraping across multiple domains
    - Track progress and maintain checkpoints
    - Apply text processing pipeline
    - Generate quality reports
    - Handle error recovery
    """
    
    def __init__(self, sample_mode: bool = False):
        """
        Initialize orchestrator.
        
        Args:
            sample_mode: If True, limit collection for testing
        """
        self.sample_mode = sample_mode
        self.start_time = datetime.now()
        self.results = {
            "news_politics": {"collected": 0, "errors": 0},
            "ecommerce_reviews": {"collected": 0, "errors": 0},
            "social_media": {"collected": 0, "errors": 0}
        }
        
        logger.info("="*70)
        logger.info("NEPALI SENTIMENT ANALYSIS - WEB SCRAPING ORCHESTRATION")
        logger.info("="*70)
        logger.info(f"Start time: {self.start_time.isoformat()}")
        logger.info(f"Sample mode: {sample_mode}")
        logger.info(f"Data directory: {DATA_DIR}")
        logger.info("")
    
    def check_environment(self) -> bool:
        """Verify all dependencies and environment setup."""
        logger.info("Checking environment and dependencies...")
        
        deps = verify_dependencies()
        all_ready = all(deps.values())
        
        logger.info("Dependency Check:")
        for package, status in deps.items():
            status_icon = "✓" if status else "✗"
            logger.info(f"  {status_icon} {package}")
        
        if not all_ready:
            missing = [p for p, s in deps.items() if not s]
            logger.error(f"Missing packages: {', '.join(missing)}")
            logger.error("Install with: pip install -r requirements.txt")
            return False
        
        logger.info("Environment check passed!\n")
        return True
    
    def scrape_news_politics_domain(self) -> int:
        """Collect news and politics content from Ratopati, Setopati, Onlinekhabar."""
        logger.info("="*70)
        logger.info("DOMAIN: NEWS & POLITICS")
        logger.info("="*70)
        
        total_collected = 0
        target = DOMAIN_TARGETS["news_politics"]["raw_target"]
        
        if self.sample_mode:
            target = 10
        
        # Instantiate scrapers
        scrapers_config = [
            ("Ratopati", RatopatiScraper(PLATFORM_CONFIG["ratopati"]), 5),
            ("Setopati", SetopatiScraper(PLATFORM_CONFIG["setopati"]), 5),
            ("Onlinekhabar", OnlineKhabarScraper(PLATFORM_CONFIG["onlinekhabar"]), 5),
        ]
        
        for platform_name, scraper, max_pages in scrapers_config:
            logger.info(f"\nScraping {platform_name}...")
            
            try:
                scraper.scrape_articles(max_pages=max_pages)
                collected = len(scraper.collected_data)
                errors = len(scraper.errors)
                
                total_collected += collected
                
                logger.info(f"✓ {platform_name}: {collected} articles, {errors} errors")
                self.results["news_politics"]["collected"] += collected
                self.results["news_politics"]["errors"] += errors
                
                # Save checkpoint
                scraper.save_checkpoint(OUTPUT_FILES["raw_corpus"])
            
            except Exception as e:
                logger.error(f"✗ {platform_name} scraping failed: {str(e)}")
                self.results["news_politics"]["errors"] += 1
        
        logger.info(f"\nNews & Politics Domain Total: {total_collected} records")
        return total_collected
    
    def scrape_ecommerce_domain(self) -> int:
        """Collect product reviews from Daraz Nepal."""
        logger.info("\n" + "="*70)
        logger.info("DOMAIN: E-COMMERCE & PRODUCT REVIEWS")
        logger.info("="*70)
        
        target = DOMAIN_TARGETS["ecommerce_reviews"]["raw_target"]
        
        if self.sample_mode:
            target = 10
        
        logger.info(f"Scraping Daraz (target: {target} products)...\n")
        
        try:
            scraper = DarazScraper(PLATFORM_CONFIG["daraz"])
            scraper.scrape_product_reviews(max_products=target)
            
            collected = len(scraper.collected_data)
            errors = len(scraper.errors)
            
            logger.info(f"✓ Daraz: {collected} reviews, {errors} errors")
            self.results["ecommerce_reviews"]["collected"] += collected
            self.results["ecommerce_reviews"]["errors"] += errors
            
            # Save checkpoint
            scraper.save_checkpoint(OUTPUT_FILES["raw_corpus"])
        
        except Exception as e:
            logger.error(f"✗ Daraz scraping failed: {str(e)}")
            self.results["ecommerce_reviews"]["errors"] += 1
            collected = 0
        
        logger.info(f"E-commerce Domain Total: {collected} records")
        return collected
    
    def scrape_social_media_domain(self, twitter_bearer_token: str = None) -> int:
        """Collect tweets from Twitter/X (Nepali language)."""
        logger.info("\n" + "="*70)
        logger.info("DOMAIN: SOCIAL & CONVERSATIONAL MEDIA")
        logger.info("="*70)
        
        target = DOMAIN_TARGETS["social_media"]["raw_target"]
        
        if self.sample_mode:
            target = 10
        
        logger.info(f"Scraping Twitter/X Nepali tweets (target: {target})...\n")
        
        if not twitter_bearer_token:
            logger.warning("⚠ Twitter API Bearer Token not provided.")
            logger.warning("  To enable Twitter scraping, set TWITTER_BEARER_TOKEN environment variable.")
            logger.warning("  Get it from: https://developer.twitter.com/en/portal/dashboard\n")
            return 0
        
        try:
            scraper = TwitterScraper(
                PLATFORM_CONFIG["twitter"],
                bearer_token=twitter_bearer_token
            )
            scraper.scrape_nepali_tweets(
                query="lang:ne",
                max_results=target
            )
            
            collected = len(scraper.collected_data)
            errors = len(scraper.errors)
            
            logger.info(f"✓ Twitter: {collected} tweets, {errors} errors")
            self.results["social_media"]["collected"] += collected
            self.results["social_media"]["errors"] += errors
            
            # Save checkpoint
            scraper.save_checkpoint(OUTPUT_FILES["raw_corpus"])
        
        except Exception as e:
            logger.error(f"✗ Twitter scraping failed: {str(e)}")
            self.results["social_media"]["errors"] += 1
            collected = 0
        
        logger.info(f"Social Media Domain Total: {collected} records")
        return collected
    
    def process_raw_corpus(self) -> int:
        """
        Apply text processing pipeline to raw collected data.
        
        Returns:
            Number of cleaned sentences
        """
        logger.info("\n" + "="*70)
        logger.info("TEXT PROCESSING PIPELINE")
        logger.info("="*70 + "\n")
        
        if not OUTPUT_FILES["raw_corpus"].exists():
            logger.error("✗ Raw corpus file not found. Run scraping first.")
            return 0
        
        processor = NepaliTextProcessor({
            "min_length": 15,
            "max_length": 350
        })
        
        sentence_count = 0
        record_count = 0
        
        logger.info("Processing raw corpus...")
        
        with open(OUTPUT_FILES["raw_corpus"], 'r', encoding='utf-8') as inf:
            with open(OUTPUT_FILES["cleaned_corpus"], 'w', encoding='utf-8') as outf:
                for line in inf:
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                        record_count += 1
                        
                        # Extract text based on domain
                        domain = record.get('domain', 'unknown')
                        
                        if domain == "news_politics":
                            text = record.get('content', '') or record.get('headline', '')
                        elif domain == "ecommerce_reviews":
                            text = record.get('review_text', '')
                        elif domain == "social_media":
                            text = record.get('text', '')
                        else:
                            text = record.get('content', '') or record.get('text', '')
                        
                        if not text:
                            continue
                        
                        # Validate Nepali content
                        if not is_valid_nepali_text(text):
                            continue
                        
                        # Process into sentences
                        sentences = processor.segment_sentences(text)
                        
                        for sentence in sentences:
                            cleaned_record = {
                                "domain": domain,
                                "platform": record.get('platform', 'unknown'),
                                "sentence": sentence,
                                "source_url": record.get('url', ''),
                                "processed_at": datetime.now().isoformat()
                            }
                            outf.write(json.dumps(cleaned_record, ensure_ascii=False) + '\n')
                            sentence_count += 1
                    
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"✓ Processed {record_count} raw records → {sentence_count} cleaned sentences")
        
        # Print statistics
        stats = processor.get_statistics([])  # Using with processed data would require rereading
        logger.info(f"  Average sentence length: ~{stats.get('avg_length', 0):.0f} characters")
        
        return sentence_count
    
    def generate_reports(self):
        """Generate comprehensive collection and quality reports."""
        logger.info("\n" + "="*70)
        logger.info("REPORT GENERATION")
        logger.info("="*70 + "\n")
        
        # Raw corpus report
        if OUTPUT_FILES["raw_corpus"].exists():
            logger.info("Raw Corpus Statistics:")
            validation = validate_jsonl_file(OUTPUT_FILES["raw_corpus"])
            logger.info(f"  Valid records: {validation['valid_lines']}")
            logger.info(f"  Invalid records: {validation['invalid_lines']}")
            
            balance_report = generate_domain_balance_report(OUTPUT_FILES["raw_corpus"])
            print_report(balance_report, "Domain Distribution")
        
        # Cleaned corpus report
        if OUTPUT_FILES["cleaned_corpus"].exists():
            logger.info("Cleaned Corpus Statistics:")
            validation = validate_jsonl_file(OUTPUT_FILES["cleaned_corpus"])
            cleaned_count = validation['valid_lines']
            logger.info(f"  Valid sentences: {cleaned_count}")
            
            balance_report = generate_domain_balance_report(OUTPUT_FILES["cleaned_corpus"])
            print_report(balance_report, "Cleaned Dataset Domain Distribution")
        
        # Summary report
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() / 60
        
        logger.info("\n" + "="*70)
        logger.info("SCRAPING SESSION SUMMARY")
        logger.info("="*70)
        logger.info(f"Duration: {duration:.1f} minutes")
        logger.info(f"Start: {self.start_time.isoformat()}")
        logger.info(f"End: {end_time.isoformat()}")
        logger.info("")
        
        for domain, stats in self.results.items():
            logger.info(f"{domain}:")
            logger.info(f"  Collected: {stats['collected']}")
            logger.info(f"  Errors: {stats['errors']}")
        
        total_collected = sum(s['collected'] for s in self.results.values())
        logger.info(f"\nTotal records collected: {total_collected}")
        logger.info("="*70)
    
    def run_full_pipeline(self, twitter_token: str = None):
        """Execute complete scraping and processing pipeline."""
        try:
            # Environment check
            if not self.check_environment():
                return False
            
            # Load environment variables (for API tokens, etc.)
            env_vars = load_environment_variables()
            
            # Use provided token or from environment
            if not twitter_token:
                twitter_token = env_vars.get('TWITTER_BEARER_TOKEN')
            
            # Execute domain-specific scraping
            logger.info("Starting multi-domain collection...\n")
            self.scrape_news_politics_domain()
            self.scrape_ecommerce_domain()
            self.scrape_social_media_domain(twitter_token)
            
            # Apply text processing pipeline
            self.process_raw_corpus()
            
            # Generate reports
            self.generate_reports()
            
            logger.info("\n✓ Scraping pipeline completed successfully!")
            return True
        
        except KeyboardInterrupt:
            logger.warning("\n⚠ Scraping interrupted by user!")
            self.generate_reports()
            return False
        
        except Exception as e:
            logger.error(f"\n✗ Fatal error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Nepali Sentiment Analysis - Multi-Domain Web Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scraper_main.py

  # Run in sample mode (quick test)
  python scraper_main.py --sample

  # Provide Twitter API token
  python scraper_main.py --twitter-token YOUR_BEARER_TOKEN
        """
    )
    
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run in sample mode (limit to 10 items per domain for testing)"
    )
    
    parser.add_argument(
        "--twitter-token",
        type=str,
        default=None,
        help="Twitter API v2 Bearer token (or set TWITTER_BEARER_TOKEN env var)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Run orchestrator
    orchestrator = NepaliCorpusOrchestrator(sample_mode=args.sample)
    success = orchestrator.run_full_pipeline(twitter_token=args.twitter_token)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
