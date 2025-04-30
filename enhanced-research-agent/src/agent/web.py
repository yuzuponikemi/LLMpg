"""
Web interaction utilities for the research agent
"""

class WebUtils:
    """Handles search results and URL browsing functionalities"""    
    
    @staticmethod
    def extract_urls_from_search_results(search_results):
        """Extract URLs from search results for auto-browsing."""
        urls = []
        
        # Handle string format from search_duckduckgo
        if isinstance(search_results, str):
            import re
            # Extract URLs using regex
            url_matches = re.findall(r'URL: (https?://[^\s\n]+)', search_results)
            title_matches = re.findall(r'Title: ([^\n]+)', search_results)
            snippet_matches = re.findall(r'Snippet: ([^\n]+(?:\n[^U][^\n]+)*)', search_results)
            
            # Create url_data objects from matches
            for i in range(len(url_matches)):
                if i < len(title_matches):
                    title = title_matches[i]
                else:
                    title = "Unknown"
                    
                if i < len(snippet_matches):
                    description = snippet_matches[i]
                else:
                    description = ""
                    
                urls.append({
                    'url': url_matches[i],
                    'title': title,
                    'description': description
                })
        
        # Also handle the original list format for backward compatibility
        elif isinstance(search_results, list):
            for result in search_results:
                if isinstance(result, dict) and 'href' in result:
                    urls.append({
                        'url': result['href'],
                        'title': result.get('title', 'Unknown'),
                        'description': result.get('body', '')
                    })
                    
        return urls
    
    @staticmethod
    def is_url_relevant(url_data, query):
        """Determine if a URL is relevant to the user's query."""
        if not url_data or not query:
            return False
            
        url = url_data['url'].lower()
        title = url_data['title'].lower()
        description = url_data['description'].lower()
        query_terms = query.lower().split()
        
        # Domain relevance check
        domains = ['github.com', 'stackoverflow.com', 'docs.', 'documentation', 
                  'arxiv.org', 'research', 'journal', 'wiki', '.edu', '.org']
        if any(domain in url for domain in domains):
            return True
            
        # Check for coding/technical keywords
        tech_indicators = ['code', 'implementation', 'example', 'tutorial', 'api', 
                         'library', 'function', 'class', 'method', 'algorithm']
        if any(indicator in url or indicator in title or indicator in description 
               for indicator in tech_indicators):
            return True
                
        # Check for science/research keywords
        science_indicators = ['paper', 'study', 'research', 'experiment', 'data', 
                            'analysis', 'results', 'methodology']
        if any(indicator in url or indicator in title or indicator in description 
               for indicator in science_indicators):
            return True
            
        return False
