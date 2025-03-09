from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import date
import requests
import logging
import time
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SlackExtractorError(Exception):
    """Base exception for errors in the Slack extractor client."""

    def __init__(self, status_code: int, error_message: str) -> None:
        self.status_code = status_code
        self.error_message = error_message
        super().__init__(
            f"Slack extractor API error. Status code: {status_code}.\n"
            f"Error: {error_message}"
        )


class DownloadError(SlackExtractorError):
    """Exception raised for errors in the download process."""
    pass


class ExtractError(SlackExtractorError):
    """Exception raised for errors in the extract process."""
    pass


class AuthenticationError(SlackExtractorError):
    """Exception raised for authentication errors."""
    pass


# Pydantic models for requests and responses
class DownloadRequest(BaseModel):
    user_id: str = Field(..., description="Slack user ID to download messages for")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")


class DownloadResponse(BaseModel):
    status: str = Field(..., description="Status of the download operation")
    message_count: int = Field(..., description="Number of messages downloaded")
    job_id: str = Field(..., description="Unique identifier for this download job")
    download_location: str = Field(..., description="Where the downloaded data is stored")


class ExtractRequest(BaseModel):
    job_id: str = Field(..., description="Job ID from a previous download operation")
    start_date: Optional[str] = Field(None, description="Optional start date to filter messages")
    end_date: Optional[str] = Field(None, description="Optional end date to filter messages")


class ExtractResponse(BaseModel):
    status: str = Field(..., description="Status of the extraction operation")
    extracted_message_count: int = Field(..., description="Number of messages extracted")
    output_file_url: Optional[str] = Field(None, description="URL to download the extracted messages")
    output_content_url: Optional[str] = Field(None, description="URL to access the extracted messages as JSON data")
    messages: Optional[List[str]] = Field(None, description="Extracted messages if inline response")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status of the service")
    version: str = Field(..., description="Version of the service")


class SlackExtractor:
    """Client for interacting with the Slack Extractor API."""
    
    def __init__(
        self, 
        base_url: str | None, 
        api_password: str | None, 
        vercel_bypass_secret: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 60
    ) -> None:
        """
        Initialize the Slack Extractor client.
        
        Args:
            base_url: Base URL of the Slack Extractor API (e.g., 'https://your-app.vercel.app/api')
            api_password: API password for authentication
            vercel_bypass_secret: Optional Vercel protection bypass secret for Vercel deployments
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
            timeout: Request timeout in seconds
        """
        if not base_url or not api_password:
            raise ValueError("Slack Extractor base URL and API password are required.")
        
        self.base_url = base_url.rstrip('/')
        self.api_password = api_password
        self.vercel_bypass_secret = vercel_bypass_secret
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
    
    def _make_request(self, method: str, endpoint: str, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Make an authenticated request to the Slack Extractor API with retry mechanism.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint (e.g., '/api/download')
            data: Request data
            
        Returns:
            Response data as dictionary
            
        Raises:
            AuthenticationError: If authentication fails
            SlackExtractorError: If the API returns an error
        """
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "X-API-Password": self.api_password
        }
        
        # Add Vercel protection bypass header - always include it
        # Use empty string as fallback if vercel_bypass_secret is None
        headers["x-vercel-protection-bypass"] = self.vercel_bypass_secret or ""
        
        # Log request details
        logger.info(f"Sending {method} request to {url}")
        logger.info(f"Headers: {headers}")
        if data:
            logger.info(f"Request data: {data}")
        
        last_exception = None
        
        # Implement retry logic
        for attempt in range(1, self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, headers=headers, timeout=self.timeout)
                else:  # POST
                    response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
                
                # Log response status and content
                logger.info(f"Response status code: {response.status_code}")
                logger.info(f"Response content: {response.text[:500]}...")  # Truncate long responses
                
                # Check for timeout errors (504, 503, etc.)
                if response.status_code in (504, 503, 502):
                    error_msg = f"Server timeout (attempt {attempt}/{self.max_retries}): {response.text}"
                    logger.warning(error_msg)
                    
                    # If we have more retries, continue to the next attempt
                    if attempt < self.max_retries:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        # If this was our last attempt, raise the error
                        raise SlackExtractorError(response.status_code, response.text)
                
                # Check for authentication errors
                if response.status_code == 401:
                    raise AuthenticationError(response.status_code, response.text)
                
                # Check for other errors
                if response.status_code != 200:
                    raise SlackExtractorError(response.status_code, response.text)
                
                return response.json()
            
            except requests.RequestException as e:
                last_exception = e
                logger.error(f"Request failed (attempt {attempt}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Max retries exceeded")
                    raise SlackExtractorError(500, f"Request failed after {self.max_retries} attempts: {str(e)}")
        
        # This should never be reached due to the raise in the loop, but just in case
        if last_exception:
            raise SlackExtractorError(500, f"Request failed: {str(last_exception)}")
        else:
            raise SlackExtractorError(500, "Request failed for unknown reason")
    
    def download_messages(
        self, user_id: str, start_date: str | date, end_date: str | date
    ) -> DownloadResponse:
        """
        Download Slack messages for a specified user within a date range.
        
        Args:
            user_id: Slack user ID to download messages for
            start_date: Start date (YYYY-MM-DD string or date object)
            end_date: End date (YYYY-MM-DD string or date object)
            
        Returns:
            DownloadResponse object with job_id and message count
            
        Raises:
            DownloadError: If the download fails
        """
        # Convert date objects to strings if needed
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
        if isinstance(end_date, date):
            end_date = end_date.isoformat()
        
        request_data = DownloadRequest(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        ).dict()
        
        try:
            response_data = self._make_request("POST", "/api/download", request_data)
            return DownloadResponse(**response_data)
        except SlackExtractorError as e:
            raise DownloadError(e.status_code, e.error_message)
    
    def extract_messages(
        self, job_id: str, start_date: Optional[str | date] = None, end_date: Optional[str | date] = None
    ) -> ExtractResponse:
        """
        Extract and format previously downloaded messages, optionally filtering by date.
        
        Args:
            job_id: Job ID from a previous download operation
            start_date: Optional start date to filter messages (YYYY-MM-DD string or date object)
            end_date: Optional end date to filter messages (YYYY-MM-DD string or date object)
            
        Returns:
            ExtractResponse object with extracted messages or file URLs
            
        Raises:
            ExtractError: If the extraction fails
        """
        # Convert date objects to strings if needed
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
        if isinstance(end_date, date):
            end_date = end_date.isoformat()
        
        request_data = ExtractRequest(
            job_id=job_id,
            start_date=start_date,
            end_date=end_date
        ).dict(exclude_none=True)
        
        try:
            response_data = self._make_request("POST", "/api/extract", request_data)
            return ExtractResponse(**response_data)
        except SlackExtractorError as e:
            raise ExtractError(e.status_code, e.error_message)
    
    def get_health(self) -> HealthResponse:
        """
        Check the health of the Slack Extractor API.
        
        Returns:
            HealthResponse object with status and version
            
        Raises:
            SlackExtractorError: If the health check fails
        """
        response_data = self._make_request("GET", "/api/health")
        return HealthResponse(**response_data)
    
    def get_message_content(self, extract_response: ExtractResponse) -> List[Dict[str, Any]]:
        """
        Get the extracted message content from an ExtractResponse.
        If messages are included in the response, returns them directly.
        Otherwise, fetches them from the output_content_url.
        
        Args:
            extract_response: ExtractResponse from a previous extract_messages call
            
        Returns:
            List of extracted messages
            
        Raises:
            ExtractError: If fetching the message content fails
        """
        # If messages are included in the response, return them
        if extract_response.messages:
            return extract_response.messages
        
        # Otherwise, fetch from the content URL
        if extract_response.output_content_url:
            try:
                # The output_content_url might already include /api, so check first
                content_url = extract_response.output_content_url
                if not content_url.startswith("/api"):
                    content_url = f"/api{content_url}"
                    
                url = f"{self.base_url}{content_url}"
                headers = {"X-API-Password": self.api_password}
                response = requests.get(url, headers=headers, timeout=self.timeout)
                
                if response.status_code != 200:
                    raise ExtractError(response.status_code, response.text)
                
                return response.json()
            except requests.RequestException as e:
                raise ExtractError(500, f"Failed to fetch message content: {str(e)}")
        
        # If no messages or content URL, return empty list
        return []


if __name__ == "__main__":
    import os
    import argparse
    from datetime import datetime, timedelta
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the Slack Extractor client")
    parser.add_argument("--base-url", default="http://localhost:8000/api", help="Base URL of the Slack Extractor API")
    parser.add_argument("--user-id", help="Slack user ID to download messages for")
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back for messages")
    parser.add_argument("--vercel-bypass", help="Vercel protection bypass secret for Vercel deployments")
    args = parser.parse_args()
    
    # Get API password from environment
    api_password = os.getenv("API_PASSWORD")
    if not api_password:
        print("Error: API_PASSWORD environment variable is not set")
        exit(1)
    
    # Get Vercel bypass secret from args or environment
    vercel_bypass_secret = args.vercel_bypass or os.getenv("VERCEL_AUTOMATION_BYPASS_SECRET")
    
    # Initialize the client
    try:
        client = SlackExtractor(args.base_url, api_password, vercel_bypass_secret)
        print(f"Initialized Slack Extractor client with base URL: {args.base_url}")
        
        # Check health
        print("\nChecking API health...")
        health = client.get_health()
        print(f"API Status: {health.status}, Version: {health.version}")
        
        # If user ID is provided, download and extract messages
        if args.user_id:
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=args.days)
            
            print(f"\nDownloading messages for user {args.user_id} from {start_date} to {end_date}...")
            download_response = client.download_messages(args.user_id, start_date, end_date)
            print(f"Download successful! Job ID: {download_response.job_id}")
            print(f"Downloaded {download_response.message_count} messages")
            
            print(f"\nExtracting messages for job {download_response.job_id}...")
            extract_response = client.extract_messages(download_response.job_id)
            print(f"Extraction successful! {extract_response.extracted_message_count} messages extracted")
            
            # Get message content
            print("\nGetting message content...")
            messages = client.get_message_content(extract_response)
            
            # Print first few messages
            if messages:
                print(f"\nFirst {min(3, len(messages))} messages:")
                for i, message in enumerate(messages[:3]):
                    print(f"\n--- Message {i+1} ---")
                    print(message)
            else:
                print("No messages found")
        else:
            print("\nNo user ID provided. Run with --user-id to download and extract messages.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
