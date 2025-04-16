import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union

import httpx
from kaggle.api.kaggle_api_extended import KaggleApi

from config.settings import KAGGLE_USERNAME, KAGGLE_KEY, HUGGINGFACE_API_KEY, DATASET_CONFIG

logger = logging.getLogger(__name__)

class DatasetTools:
    """Tool for managing datasets."""

    def __init__(self):
        """Initialize the DatasetTools tool."""
        self.sources = DATASET_CONFIG["sources"]
        self.max_results_per_source = DATASET_CONFIG["max_results_per_source"]
        self.timeout = DATASET_CONFIG["timeout"]

        # Set up Kaggle credentials
        if KAGGLE_USERNAME and KAGGLE_KEY:
            os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
            os.environ['KAGGLE_KEY'] = KAGGLE_KEY
            self.kaggle_api = True
            try:
                self.kaggle_api = KaggleApi()
                self.kaggle_api.authenticate()
            except Exception as e:
                logger.error(f"Failed to authenticate with Kaggle API: {e}")
                self.kaggle_api = False
        else:
            logger.warning("Kaggle credentials not found. Kaggle API will not be available.")
            self.kaggle_api = False

        # Check if HuggingFace credentials are set
        self.huggingface_api_key = HUGGINGFACE_API_KEY
        self.huggingface_api_available = bool(self.huggingface_api_key)
        if not self.huggingface_api_available:
            logger.warning("HuggingFace API key not found. HuggingFace API will not be available.")

    async def search_kaggle_datasets(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search Kaggle datasets based on a query.
        Args:
            query: The search query.
            max_results: The maximum number of results to return.
            Returns:
            List of Kaggle datasets matching the query.
        """

        max_results = max_results or self.max_results_per_source

        if not self.kaggle_api:
            logger.warning("Kaggle API is not available. Skipping Kaggle dataset search.")
            return []
        try:
            logger.info(f"Searching Kaggle datasets for query: {query}")

            # Search for datasets
            datasets = await asyncio.to_thread(
                self.kaggle_api.datasets_list,
                search=query,
                sort_by="relevance",
                max_size=None,
                file_type=None,
                license_name=None,
                page_size=max_results
                )

            # Extract dataset information
            results = []
            for dataset in datasets:
                results.append({
                    "title": dataset.title,
                    "name": dataset.ref,
                    "url": f"https://www.kaggle.com/datasets/{dataset.ref}",
                    "description": dataset.description,
                    "source": "kaggle",
                    "size": dataset.size,
                    "last_updated": dataset.lastUpdated,
                    "download_count": dataset.downloadCount
                })

            logger.info(f"Found {len(results)} Kaggle datasets for query: {query}")
            return results[:max_results]
        
        except Exception as e:
            logger.error(f"Error searching Kaggle datasets: {e}")
            return []
    
    async def search_huggingface_datasets(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search HuggingFace datasets based on a query.
        Args:
            query: The search query.
            max_results: The maximum number of results to

            Returns:
            List of HuggingFace datasets matching the query.
        """

        max_results = max_results or self.max_results_per_source

        if not self.huggingface_api_available:
            logger.warning("HuggingFace API is not available. Skipping HuggingFace dataset search.")
            return []
        try:
            logger.info(f"Searching HuggingFace datasets for query: {query}")

            # Search HuggingFace API
            headers = {}
            if self.huggingface_api_key:
                headers["Authorization"] = f"Bearer {self.huggingface_api_key}"
                
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    "https://huggingface.co/api/datasets",
                    params={"search": query, "limit": max_results},
                    headers=headers
                )
                response.raise_for_status()
                datasets = response.json()
            
            # Extract dataset information
            results = []
            for dataset in datasets:
                results.append({
                    "title": dataset.get("id", ""),
                    "name": dataset.get("id", ""),
                    "url": f"https://huggingface.co/datasets/{dataset.get('id', '')}",
                    "description": dataset.get("description", ""),
                    "source": "huggingface",
                    "author": dataset.get("author", ""),
                    "tags": dataset.get("tags", [])
                })
            
            logger.info(f"Found {len(results)} HuggingFace datasets for query: {query}")
            return results[:max_results]
        
        except Exception as e:
            logger.error(f"Error searching HuggingFace datasets: {e}")
            return []
        
    async def search_datasets(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search datasets from multiple sources based on a query.
        Args:
            query: The search query.
            max_results: The maximum number of results to return.

            Returns:
            List of datasets matching the query.
        """

        max_results = max_results or self.max_results_per_source

        try:
            logger.info(f"Searching GitHub for datasets: {query}")
            
            # Search GitHub API
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    "https://api.github.com/search/repositories",
                    params={"q": f"{query} dataset", "sort": "stars", "order": "desc", "per_page": max_results}
                )
                response.raise_for_status()
                data = response.json()
                
                repositories = data.get("items", [])
            
            # Extract repository information
            results = []
            for repo in repositories:
                results.append({
                    "title": repo.get("name", ""),
                    "name": repo.get("full_name", ""),
                    "url": repo.get("html_url", ""),
                    "description": repo.get("description", ""),
                    "source": "github",
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "last_updated": repo.get("updated_at", "")
                })
            
            logger.info(f"Found {len(results)} GitHub repositories for query: {query}")
            return results[:max_results]
        
        except Exception as e:
            logger.error(f"Error searching GitHub repositories: {e}")
            return []
    
    async def search_datasets(self, query: str, sources: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Search for datasets across multiple sources.
        
        Args:
            query: The search query.
            sources: Optional list of sources to search.
            
        Returns:
            Dictionary mapping source names to lists of dataset information.
        """
        # Filter unavailable sources
        available_sources = []
        if "kaggle" in sources and self.kaggle_available:
            available_sources.append("kaggle")
        if "huggingface" in sources and self.huggingface_available:
            available_sources.append("huggingface")
        if "github" in sources:
            available_sources.append("github")
            
        if not available_sources:
            logger.warning("No available dataset sources found")
            return {}
            
        sources = available_sources
        results = {}
        
        search_coroutines = []
        
        # Add search coroutines for each source
        if "kaggle" in sources:
            search_coroutines.append(self.search_kaggle_datasets(query))
        
        if "huggingface" in sources:
            search_coroutines.append(self.search_huggingface_datasets(query))
        
        if "github" in sources:
            search_coroutines.append(self.search_github_datasets(query))
        
        # Run searches in parallel
        all_results = await asyncio.gather(*search_coroutines, return_exceptions=True)
        
        # Process results
        result_index = 0
        if "kaggle" in sources:
            if isinstance(all_results[result_index], Exception):
                logger.error(f"Error searching Kaggle: {all_results[result_index]}")
                results["kaggle"] = []
            else:
                results["kaggle"] = all_results[result_index]
            result_index += 1
        
        if "huggingface" in sources:
            if isinstance(all_results[result_index], Exception):
                logger.error(f"Error searching HuggingFace: {all_results[result_index]}")
                results["huggingface"] = []
            else:
                results["huggingface"] = all_results[result_index]
            result_index += 1
        
        if "github" in sources:
            if isinstance(all_results[result_index], Exception):
                logger.error(f"Error searching GitHub: {all_results[result_index]}")
                results["github"] = []
            else:
                results["github"] = all_results[result_index]
            result_index += 1
        
        return results
    
    async def find_datasets_for_use_case(self, use_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find datasets relevant to a specific use case.
        
        Args:
            use_case: Dictionary describing the use case.
            
        Returns:
            List of relevant datasets.
        """
        title = use_case.get("title", "")
        description = use_case.get("description", "")
        
        # Generate search queries based on use case
        queries = [
            title,
            f"{title} dataset",
            f"{title} machine learning"
        ]
        
        if "industry" in use_case:
            queries.append(f"{use_case['industry']} {title} dataset")
        
        if "keywords" in use_case and isinstance(use_case["keywords"], list):
            for keyword in use_case["keywords"][:2]:  # Limit to top 2 keywords
                queries.append(f"{keyword} dataset")
        
        # Search for datasets using each query
        all_datasets = []
        for query in queries:
            source_results = await self.search_datasets(query)
            
            # Flatten results from all sources
            for source, datasets in source_results.items():
                for dataset in datasets:
                    # Add the query that found this dataset
                    dataset["query"] = query
                    all_datasets.append(dataset)
        
        # Deduplicate datasets based on URL
        unique_datasets = {}
        for dataset in all_datasets:
            url = dataset.get("url", "")
            if url and url not in unique_datasets:
                unique_datasets[url] = dataset
        
        return list(unique_datasets.values())
