import re
import asyncio
import base64
import httpx
import json
from pathlib import Path
from typing import Optional, List, Tuple
from openai import OpenAI
import os
import tempfile
import aiohttp
from urllib.parse import urlparse
from fastapi import Depends, HTTPException, Request, status

from src.core.utils.logger.custom_logging import LoggerMixin
from src.core.schemas.menu import *
from src.core.providers.provider_factory import ModelProviderFactory, ProviderType
from src.core.utils.config import settings
from src.core.providers.base_provider import ModelProvider

class MenuExtractionHandler(LoggerMixin):
    """Handler for extracting menu information from images using LLMs"""
    
    def __init__(self):
        super().__init__()
        self._provider_instances = {}  # Cache for provider instances
           
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_system_prompt(self) -> str:
        """System prompt specifically designed for restaurant menu extraction"""
        return """You are a professional menu data extraction specialist. Your job is to carefully read every detail in the menu image and extract complete information for each dish.

    CRITICAL: Look at the image very carefully. Most menus show:
    - Dish name
    - Price 
    - Description/ingredients (usually on the line below the dish name)

    EXTRACTION RULES:
    1. Read EVERY line of text in the image
    2. For each dish, look for:
    - Main dish name (usually bold or prominent)
    - Price (number, often at the end of the dish name line)
    - Description (usually smaller text below the dish name - THIS IS VERY IMPORTANT)
    3. Do NOT skip or ignore any text you see
    4. Include ALL ingredients, cooking methods, and descriptions exactly as written

    OUTPUT FORMAT - Use this EXACT structure:

    ### 1. [Exact Dish Name]
    **Price**: [number only]
    **Currency**: USD
    **Description**: [Complete description including all ingredients and cooking methods you can see]
    **Category**: [Appetizers/Entrees/etc]

    ### 2. [Next Dish Name]  
    **Price**: [number only]
    **Currency**: USD
    **Description**: [Complete description - never leave this empty if you see description text]
    **Category**: [category]

    EXAMPLE based on typical Italian menu:
    ### 1. Bruschetta all'Italiana
    **Price**: 9
    **Currency**: USD  
    **Description**: Grilled bread, tomatoes, basil, garlic & extra virgin olive oil
    **Category**: Appetizers

    ### 2. Impepata di Cozze
    **Price**: 15
    **Currency**: USD
    **Description**: Mussels in a white wine and cherry tomato broth, served with homemade garlic crostini  
    **Category**: Appetizers

    IMPORTANT REMINDERS:
    - READ THE IMAGE CAREFULLY - there is usually description text below each dish name
    - NEVER write "No description visible" if you can see any text below the dish name
    - Include EVERY ingredient and detail you can read
    - The description is usually the smaller text under the main dish name
    - Extract information for EVERY dish you can see in the image
    - Preserve original language and spelling exactly as shown"""

    def _get_json_conversion_prompt(self) -> str:
        """System prompt for converting markdown to JSON format"""
        return """You are a data format converter. Your task is to convert the markdown menu data into a properly structured JSON array.

CONVERSION RULES:
1. Convert ALL menu items from the markdown into JSON objects
2. Each item must have these exact fields:
   - "name": string (dish name)
   - "price": number (numeric value only, use 0 if "Not listed" or not available)
   - "currency": string (currency code like "USD", "VND", etc. Use null if not available)
   - "description": string (full description, use empty string "" if not available)
   - "category": string (category like "Appetizers", "Entrees", etc. Use empty string "" if not available)

3. Price field must be a NUMBER, not a string
4. If price shows "Not listed" or is missing, use 0
5. Preserve all original text exactly as shown in the markdown
6. Do not add or modify any information

OUTPUT MUST BE VALID JSON ARRAY:
[
  {
    "name": "Dish Name",
    "price": 15.99,
    "currency": "USD",
    "description": "Complete description with ingredients",
    "category": "Appetizers"
  },
  {
    "name": "Another Dish",
    "price": 0,
    "currency": null,
    "description": "",
    "category": "Entrees"
  }
]

CRITICAL: Return ONLY the JSON array, no additional text or markdown formatting."""

    async def _extract_with_provider(self, image_path: str, provider_type: str, 
                                   model_name: str, api_key: Optional[str] = None) -> str:
        """Extract menu using the specified provider"""
        base64_image = self._encode_image(image_path)
        
        if provider_type == ProviderType.OPENAI:
            return await self._extract_with_openai(base64_image, model_name, api_key)
        elif provider_type == ProviderType.OLLAMA:
            return await self._extract_with_ollama(base64_image, model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")

    async def _convert_to_json_with_provider(self, markdown_text: str, provider_type: str,
                                           model_name: str, api_key: Optional[str] = None) -> str:
        """Convert markdown to JSON using the specified provider"""
        
        if provider_type == ProviderType.OPENAI:
            return await self._convert_to_json_openai(markdown_text, model_name, api_key)
        elif provider_type == ProviderType.OLLAMA:
            return await self._convert_to_json_ollama(markdown_text, model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")

    async def _convert_to_json_openai(self, markdown_text: str, model_name: str, api_key: str) -> str:
        """Convert markdown to JSON using OpenAI API"""
        try:
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_json_conversion_prompt()
                    },
                    {
                        "role": "user",
                        "content": f"Convert this markdown menu data to JSON format:\n\n{markdown_text}"
                    }
                ],
                max_tokens=4096,
                temperature=0.0  # Use 0 for consistent formatting
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI JSON conversion error: {str(e)}")
            raise

    async def _convert_to_json_ollama(self, markdown_text: str, model_name: str) -> str:
        """Convert markdown to JSON using Ollama local model"""
        payload = {
            "model": model_name,
            "prompt": f"{self._get_json_conversion_prompt()}\n\nConvert this markdown menu data to JSON format:\n\n{markdown_text}",
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.text}")
                
                return response.json()['response'].strip()
                
        except httpx.ConnectError:
            raise Exception("Cannot connect to Ollama. Make sure Ollama is running locally.")
        except Exception as e:
            self.logger.error(f"Ollama JSON conversion error: {str(e)}")
            raise
    
    async def _extract_with_openai(self, base64_image: str, model_name: str, api_key: str) -> str:
        """Extract using OpenAI API"""
        try:
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an expert at reading restaurant menus and extracting complete information including dish names, prices, and descriptions/ingredients."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def _extract_with_ollama(self, base64_image: str, model_name: str) -> str:
        """Extract using Ollama local model"""
        payload = {
            "model": model_name,
            "prompt": f"{self._get_system_prompt()}\n\nPlease extract all menu items from this image.",
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.text}")
                
                return response.json()['response']
                
        except httpx.ConnectError:
            raise Exception("Cannot connect to Ollama. Make sure Ollama is running locally.")
        except Exception as e:
            self.logger.error(f"Ollama error: {str(e)}")
            raise
    
    def _normalize_currency(self, currency: str) -> str:
        """Normalize currency codes to standard format"""
        if not currency:
            return None
            
        # Currency mapping
        currency_map = {
            # Vietnamese
            'đ': 'VND',
            'vnđ': 'VND',
            'vnd': 'VND',
            'đồng': 'VND',
            'dong': 'VND',
            
            # US Dollar
            '$': 'USD',
            'usd': 'USD',
            'dollar': 'USD',
            'dollars': 'USD',
            
            # Euro
            '€': 'EUR',
            'eur': 'EUR',
            'euro': 'EUR',
            
            # British Pound
            '£': 'GBP',
            'gbp': 'GBP',
            'pound': 'GBP',
            
            # Japanese Yen
            '¥': 'JPY',
            'jpy': 'JPY',
            'yen': 'JPY',
            
            # Thai Baht
            'บาท': 'THB',
            'thb': 'THB',
            'baht': 'THB',
            '฿': 'THB',
            
            # Korean Won
            '₩': 'KRW',
            'krw': 'KRW',
            'won': 'KRW',
            '원': 'KRW',
            
            # Others
            'rmb': 'CNY',
            'yuan': 'CNY',
            '元': 'CNY',
        }
        
        # Normalize and lookup
        currency_lower = currency.lower().strip()
        
        # Direct match
        if currency_lower in currency_map:
            return currency_map[currency_lower]
        
        # Check if it's already a valid code
        if len(currency) == 3 and currency.upper() == currency:
            return currency
        
        # Default to uppercase
        return currency.upper()

    def _parse_json_to_menu_items(self, json_text: str) -> List[MenuItem]:
        """Parse JSON output to structured menu items"""
        try:
            # Clean the response in case it has markdown formatting
            json_text = json_text.strip()
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            
            # Parse JSON
            items_data = json.loads(json_text)
            
            if not isinstance(items_data, list):
                raise ValueError("Expected JSON array")
            
            menu_items = []
            for item_data in items_data:
                # Validate required fields
                if not isinstance(item_data, dict):
                    self.logger.warning(f"Skipping invalid item: {item_data}")
                    continue
                
                # Create MenuItem with proper type conversion
                menu_item = MenuItem(
                    name=str(item_data.get('name', '')),
                    price=float(item_data.get('price', 0)) if item_data.get('price') not in [None, '', 'Not listed'] else None,
                    currency=self._normalize_currency(item_data.get('currency')) if item_data.get('currency') else None,
                    description=str(item_data.get('description', '')) if item_data.get('description') else None,
                    category=str(item_data.get('category', '')) if item_data.get('category') else None
                )
                
                menu_items.append(menu_item)
            
            return menu_items
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            self.logger.error(f"Raw JSON text: {json_text}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error parsing JSON to menu items: {str(e)}")
            raise

    def _parse_markdown_to_menu_items(self, markdown_text: str) -> List[MenuItem]:
        """Parse markdown output to structured menu items (fallback method)"""
        menu_items = []
        
        # Split by item headers (### followed by number)
        item_blocks = re.split(r'###\s*\d+\.?\s*', markdown_text)
        
        for block in item_blocks[1:]:  # Skip first empty split
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            
            # Extract name from first line
            name = lines[0].strip().replace('[', '').replace(']', '')
            
            # Extract price and description
            price = None
            description = None
            
            for line in lines[1:]:
                if '**Price**:' in line or '**Giá**:' in line:
                    price = line.split(':', 1)[1].strip()
                elif '**Description**:' in line or '**Mô tả**:' in line:
                    description = line.split(':', 1)[1].strip()
            
            menu_items.append(MenuItem(
                name=name,
                price=price,
                description=description
            ))
        
        return menu_items

    def _post_process_items(self, menu_items: List[MenuItem]) -> List[MenuItem]:
        """Post-process items to ensure data consistency"""
        
        # Detect dominant currency if some items missing currency
        currencies = [item.currency for item in menu_items if item.currency]
        dominant_currency = None
        
        if currencies:
            from collections import Counter
            currency_counts = Counter(currencies)
            dominant_currency = currency_counts.most_common(1)[0][0]
        
        # Process each item
        for item in menu_items:
            # If has price but no currency, use dominant currency
            if item.price and not item.currency and dominant_currency:
                item.currency = dominant_currency
                self.logger.info(f"Inferred currency {dominant_currency} for item: {item.name}")
        
        return menu_items

    async def extract_menu(
        self, 
        image_path: str,
        provider_type: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> List[MenuItem]:
        """
        Extract menu from image and return structured data
        
        Args:
            image_path: Path to the image file
            provider_type: Provider type (openai, ollama, gemini)
            model_name: Model name (optional, will use default if not provided)
            api_key: API key for paid providers (optional for ollama)
            
        Returns:
            List of menu items
        """        
        is_url = self._is_url(image_path)

        if not is_url:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            # Check file size cho local file
            file_size = Path(image_path).stat().st_size / (1024 * 1024)  # MB
            if file_size > 20:
                raise ValueError(f"Image file too large: {file_size:.1f}MB. Maximum is 20MB.")
        
        # Set default model if not provided
        if not model_name:
            raise ValueError(f"No default model for provider {provider_type}")
    
        # Extract menu using provider
        self.logger.info(f"Extracting menu using {provider_type} provider with model {model_name}")

        temp_file_path = None
        
        try:
            # Process image input
            if is_url:
                temp_file_path = await self._download_image(image_path)
                processed_image_path = temp_file_path
                
                # Kiểm tra file size sau khi download
                file_size = Path(processed_image_path).stat().st_size / (1024 * 1024)
                if file_size > 20:
                    raise ValueError(f"Downloaded image too large: {file_size:.1f}MB. Maximum is 20MB.")
            else:
                processed_image_path = image_path

            # Step 1: Extract with provider (returns markdown)
            raw_markdown = await self._extract_with_provider(
                image_path=processed_image_path,
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            self.logger.info("Step 1: Raw markdown extraction completed")
            
            # Step 2: Convert markdown to JSON using LLM
            try:
                self.logger.info("Step 2: Converting markdown to JSON format")
                json_text = await self._convert_to_json_with_provider(
                    markdown_text=raw_markdown,
                    provider_type=provider_type,
                    model_name=model_name,
                    api_key=api_key
                )
                
                # Parse JSON to menu items
                menu_items = self._parse_json_to_menu_items(json_text)
                self.logger.info(f"Successfully parsed {len(menu_items)} items from JSON")
                
            except Exception as e:
                self.logger.warning(f"JSON conversion failed: {e}, falling back to markdown parsing")
                # Fallback to original markdown parsing
                menu_items = self._parse_markdown_to_menu_items(raw_markdown)
            
            # Post-process: Ensure currency for items with price
            menu_items = self._post_process_items(menu_items)
            
            self.logger.info(f"Extracted {len(menu_items)} menu items")
            return menu_items
        
        except Exception as e:
            self.logger.error(f"Error extracting menu: {str(e)}")
            raise

        finally:
            # Cleanup temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    self.logger.info(f"Temp file deleted: {temp_file_path}")
                except Exception as e:
                    self.logger.error(f"Cannot delete temp file {temp_file_path}: {str(e)}")

    async def _process_image_input(self, image_path: str) -> str:
        """
        Input processing: if it is a URL then download, if it is a local path then keep it as is
        """
        if self._is_url(image_path):
            return await self._download_image(image_path)
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File does not exist: {image_path}")
            return image_path
    
    def _is_url(self, path: str) -> bool:
        """
        Check if path is a URL
        """
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except:
            return False
    
    async def _download_image(self, url: str) -> str:
        """
        Download image from URL với timeout và size limit
        """
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unable to download image from URL: {url}. Status: {response.status}"
                        )
                    
                    # Check content-length before downloading
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > 20:
                            raise ValueError(f"Image too large: {size_mb:.1f}MB. Maximum is 20MB.")
                    
                    # Get file extension
                    extension = self._get_file_extension(url, response.headers.get('content-type'))
                    
                    # Download with chunks to avoid loading all into memory
                    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
                        downloaded = 0
                        async for chunk in response.content.iter_chunked(8192):  # 8KB chunks
                            downloaded += len(chunk)
                            if downloaded > 20 * 1024 * 1024:  # 20MB limit
                                tmp_file.close()
                                os.remove(tmp_file.name)
                                raise ValueError("Image too large during download. Maximum is 20MB.")
                            tmp_file.write(chunk)
                        
                        temp_path = tmp_file.name
                    
                    self.logger.info(f"Downloaded image from {url} to {temp_path} ({downloaded/1024/1024:.1f}MB)")
                    return temp_path
                        
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"Timeout downloading image from URL: {url}"
            )
        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error downloading image: {str(e)}"
            )
    
    def _get_file_extension(self, url: str, content_type: Optional[str] = None) -> str:
        """
        Get file extension from URL or content-type
        """
        path = urlparse(url).path
        if '.' in path:
            return os.path.splitext(path)[1]
        
        if content_type:
            if 'jpeg' in content_type or 'jpg' in content_type:
                return '.jpg'
            elif 'png' in content_type:
                return '.png'
            elif 'webp' in content_type:
                return '.webp'
        
        # Default
        return '.jpg'
    

    # API 2: Generate Prompt
    async def generate_dish_description(
        self,
        request: DishDescriptionRequest,
        provider_type: str,
        model_name: str,
        api_key: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate detailed dish description for image generation
        
        Returns:
            Dict with 'description' and 'prompt_ready' keys
        """
        try:
            # Generate description using provider
            description = await self._generate_with_provider(
                request=request,
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            
            # Build complete prompt for ComfyUI
            prompt_ready = self._build_comfyui_prompt(description, request.cultural_style)
            
            return {
                "description": description,
                "prompt_ready": prompt_ready
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dish description: {str(e)}")
            raise
    
    async def get_provider(self,
                            model_name: str, 
                            provider_type: str = ProviderType.OPENAI, 
                            api_key: Optional[str] = None) -> ModelProvider:
            """
            Get a model provider instance for translation.
            
            Args:
                model_name: Name of the model to use
                provider_type: Type of provider (ollama, openai, gemini)
                api_key: API key for paid providers
                
            Returns:
                ModelProvider: Provider instance
            """
            try:
                # Create unique key for caching
                cache_key = f"{provider_type}:{model_name}:{api_key}"
                
                if cache_key not in self._provider_instances:
                    # Create provider
                    provider = ModelProviderFactory.create_provider(
                        provider_type=provider_type,
                        model_name=model_name,
                        api_key=api_key
                    )
                    
                    # Initialize model
                    await provider.initialize()
                    
                    # Cache provider
                    self._provider_instances[cache_key] = provider
                
                return self._provider_instances[cache_key]
                
            except Exception as e:
                self.logger.error(f"Error getting provider for translation: {str(e)}")
                raise

    async def _generate_with_provider(
        self,
        request: DishDescriptionRequest,
        provider_type: str,
        model_name: str,
        api_key: Optional[str] = None
    ) -> str:
        """Generate description using the specified provider"""
        
        if provider_type == ProviderType.OPENAI:
            # Get provider instance
            provider = await self.get_provider(
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )
            
            # Build prompts
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_user_prompt(request)
            
            # Generate description using provider
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call provider's generate method
            response = await provider.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            # Extract text from response
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'content' in response:
                return response['content']
            elif isinstance(response, str):
                return response
            else:
                raise ValueError(f"Unexpected response format from provider: {type(response)}")
            
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")
        
    
    def _get_system_prompt(self) -> str:

        return """You are a culinary expert and food photographer who creates detailed, visual descriptions of dishes for AI image generation.

    Your task is to generate a rich, detailed description that helps an AI model understand:
    1. The visual appearance of the dish
    2. Key ingredients and their arrangement
    3. Textures, colors, and presentation style
    4. Garnishes and plating techniques
    5. The overall aesthetic and appeal

    IMPORTANT GUIDELINES:
    - If a menu description is provided that mentions specific ingredients or components, you MUST include ALL those ingredients visually in your description
    - Focus on VISUAL elements that can be photographed
    - Be specific about colors, textures, and arrangements of each component mentioned
    - Describe how each ingredient mentioned in the description would appear in the photo
    - Mention typical garnishes and presentation styles
    - Keep descriptions concise but detailed (2-3 sentences)
    - Use professional culinary terminology when appropriate
    - Consider cultural authenticity in presentation

    For example: If the description mentions "grilled chicken with herbs and lemon", ensure your visual description includes the golden-brown grilled chicken, visible herb garnish (like rosemary or thyme), and lemon wedges or slices as key visual elements."""

    def _build_user_prompt(self, request: DishDescriptionRequest) -> str:
        """Build user prompt with all available information"""
        prompt_parts = [f"Generate a detailed visual description for food photography of: {request.name}"]
        
        # Add description with emphasis if available
        if request.description:
            prompt_parts.append(f"\nIMPORTANT - Menu description with ingredients: {request.description}")
            prompt_parts.append("Make sure to visually include ALL components mentioned in this description.")
        
        if request.category:
            prompt_parts.append(f"Category: {request.category}")
        
        if request.cultural_style:
            prompt_parts.append(f"Cuisine style: {request.cultural_style}")
        
        if request.country:
            prompt_parts.append(f"Note that you need to add the country to the description to know which country the dish belongs to in order to create a description that is most suitable for their culinary culture: {request.country}")
        
        # Determine presentation style based on price if available
        if request.price and request.currency:
            price_context = self._get_price_context(request.price, request.currency)
            if price_context:
                prompt_parts.append(price_context)
        
        # Final instruction
        prompt_parts.append("\nProvide a visual description focusing on appearance, ingredients, plating, and presentation style.")
        
        if request.description:
            prompt_parts.append("Remember to include visual details for every ingredient mentioned in the menu description.")
        
        return "\n".join(prompt_parts)
        
    def _get_price_context(self, price: float, currency: str) -> Optional[str]:
        """Get context based on price range"""
        # Simple price range detection
        if not currency:
            return None
            
        # Convert to relative scale
        if currency == "VND":
            if price < 50000:
                return "This is a casual, everyday dish"
            elif price < 200000:
                return "This is a mid-range restaurant dish"
            else:
                return "This is a premium, fine-dining dish"
        elif currency == "USD":
            if price < 10:
                return "This is a casual, everyday dish"
            elif price < 30:
                return "This is a mid-range restaurant dish"
            else:
                return "This is a premium, fine-dining dish"
        
        return None
    
    def _build_comfyui_prompt(self, description: str, cultural_style: Optional[CulturalStyle]) -> str:
        """Build complete prompt ready for ComfyUI"""
        cultural_str = cultural_style.value if cultural_style else ""
        
        # Build prompt components
        components = [
            f"Professional food advertising photography of {description}",
            cultural_str,
            "beautifully plated with attention to detail",
            "appetizing presentation with natural garnishes",
            "optimal lighting to enhance textures and colors",
            "clean professional background",
            "ultra-realistic render with visible fresh ingredients",
            "shot in high-end commercial style",
            "sharp focus, depth of field",
            "ready for menu or advertising use"
        ]
        
        # Filter out empty components and join
        prompt = ", ".join(filter(None, components))
        
        # Clean up extra spaces
        prompt = " ".join(prompt.split())
        
        return prompt