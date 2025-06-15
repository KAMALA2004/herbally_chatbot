import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import tweepy
from google import genai
from agno.agent import Agent
from agno.models.google import Gemini
import time
import requests
from PIL import Image
from io import BytesIO
from google.genai import types
import hashlib
import hmac
import httpx
from openai import AzureOpenAI
import firebase_admin
from firebase_admin import credentials

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("firebasekey.json")
        firebase_admin.initialize_app(cred)
        st.session_state.firebase_initialized = True
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}")
        st.session_state.firebase_initialized = False

load_dotenv()

client = genai.Client(api_key=os.getenv("TEXT_API_KEY"))

# Configure Azure OpenAI for DALL-E 3
azure_client = AzureOpenAI(
    api_version="2024-02-01",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Twitter configuration
TWITTER_API_KEY = os.getenv("CONSUMER_KEY")
TWITTER_API_SECRET = os.getenv("CONSUMER_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("ACCESS_KEY")
TWITTER_ACCESS_SECRET = os.getenv("ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# LinkedIn configuration
LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_ORG_ID = os.getenv("LINKEDIN_ORG_ID")
LINKEDIN_ORG_URN = f'urn:li:organization:{LINKEDIN_ORG_ID}'

# LinkedIn headers
linkedin_headers = {
    'Authorization': f'Bearer {LINKEDIN_ACCESS_TOKEN}',
    'Content-Type': 'application/json',
    'X-Restli-Protocol-Version': '2.0.0'
}

# Initialize Twitter API
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
twitter_api = tweepy.API(auth)

# Configure Twitter client
twitter_client = tweepy.Client(
    bearer_token=TWITTER_BEARER_TOKEN,
    consumer_key=TWITTER_API_KEY,
    consumer_secret=TWITTER_API_SECRET,
    access_token=TWITTER_ACCESS_TOKEN,
    access_token_secret=TWITTER_ACCESS_SECRET
)

# Define all available AI models
AI_MODELS = {
    "caption_agent": {
        "name": "Caption Generation Agent",
        "description": "Generate engaging social media captions for herbal lifestyle products",
        "model": "gemini-2.0-flash",
        "status": "active",
        "usage_count": 0,
        "accuracy": 0.85,
        "engagement_rate": 0.12
    },
    "image_prompt_agent": {
        "name": "Image Prompt Generation Agent", 
        "description": "Generate detailed image prompts for serene, wellness-focused product photography",
        "model": "gemini-2.0-flash",
        "status": "active",
        "usage_count": 0,
        "accuracy": 0.78,
        "engagement_rate": 0.15
    },
    "email_agent": {
        "name": "Email Generation Agent",
        "description": "Generate professional emails for the herbal lifestyle brand",
        "model": "gemini-2.0-flash", 
        "status": "active",
        "usage_count": 0,
        "accuracy": 0.82,
        "engagement_rate": 0.08
    },
    "research_agent": {
        "name": "Research Analysis Agent",
        "description": "Search and analyze social media content for the herbal lifestyle brand",
        "model": "gemini-2.0-flash",
        "status": "active", 
        "usage_count": 0,
        "accuracy": 0.79,
        "engagement_rate": 0.06
    },
    "linkedin_caption_agent": {
        "name": "LinkedIn Caption Generation Agent",
        "description": "Generate professional, business-focused social media captions for LinkedIn",
        "model": "gemini-2.0-flash",
        "status": "active",
        "usage_count": 0,
        "accuracy": 0.87,
        "engagement_rate": 0.14
    },
    "twitter_caption_agent": {
        "name": "Twitter Caption Generation Agent", 
        "description": "Generate engaging, concise social media captions for Twitter",
        "model": "gemini-2.0-flash",
        "status": "active",
        "usage_count": 0,
        "accuracy": 0.83,
        "engagement_rate": 0.16
    },
    "linkedin_image_agent": {
        "name": "LinkedIn Image Prompt Generation Agent",
        "description": "Generate professional, high-end product photography prompts for LinkedIn",
        "model": "gemini-2.0-flash",
        "status": "active",
        "usage_count": 0,
        "accuracy": 0.81,
        "engagement_rate": 0.13
    }
}

# Content genres
CONTENT_GENRES = [
    "Luxury", "Minimalistic", "Vintage", "Modern", "Bohemian", 
    "Corporate", "Casual", "Elegant", "Rustic", "Contemporary"
]

# Content types
CONTENT_TYPES = [
    "Posts", "Reels", "Stories", "Carousel", "Video", "IGTV"
]

# Platform-specific content types
PLATFORM_CONTENT_TYPES = {
    "Twitter": ["Tweet", "Thread", "Quote Tweet"],
    "LinkedIn": ["Post", "Article", "Carousel", "Video"],
    "Instagram": ["Post", "Reel", "Story", "IGTV", "Carousel"]
}

# Add color palette at the top of the file
COLOR_PALETTE = {
    "tulsi": "#a3ceb3",
    "vetiver": "#f5d897", 
    "turmeric": "#e9b345",
    "ashwagandha": "#e6a794",
    "sandalo": "#cebbab",
    "red_sandal": "#f5a770",
    "ashoka": "#c89bbe",
    "indigo": "#83a0c2",
    "aloe": "#98c1a9",
    "neem": "#7d9f7d"
}

def admin_login():
    """Admin login interface using Firebase Authentication"""
    st.markdown("### 🔐 Admin Authentication")
    
    # Initialize session state for authentication
    if "signedout" not in st.session_state:
        st.session_state["signedout"] = False
    if 'signout' not in st.session_state:
        st.session_state['signout'] = False
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''
    
    if not st.session_state["signedout"]:
        choice = st.selectbox('Choose an option:', ['Login', 'Sign up', 'Forget Password'])
        
        if choice == 'Sign up':
            st.subheader("Create Admin Account")
            email = st.text_input('Email Address')
            password = st.text_input('Password', type='password')
            username = st.text_input("Enter your unique username")
            
            if st.button('Create Admin Account'):
                if email and password and username:
                    user = sign_up_with_email_and_password(email=email, password=password, username=username)
                    if user:
                        st.success('Admin account created successfully!')
                        st.markdown('Please Login using your email and password')
                        st.balloons()
                else:
                    st.error('Please fill in all fields')
        
        elif choice == 'Login':
            st.subheader("Admin Login")
            email = st.text_input('Email Address')
            password = st.text_input('Password', type='password')
            st.session_state.email_input = email
            st.session_state.password_input = password
            
            if st.button('Login'):
                if email and password:
                    try:
                        userinfo = sign_in_with_email_and_password(email, password)
                        if userinfo:
                            st.session_state.username = userinfo['username'] or email
                            st.session_state.useremail = userinfo['email']
                            st.session_state.signedout = True
                            st.session_state.signout = True
                            st.session_state.admin_authenticated = True
                            st.session_state.admin_username = st.session_state.username
                            st.success("✅ Login successful! Welcome, Admin.")
                            st.rerun()
                        else:
                            st.error("❌ Login failed. Please check your credentials.")
                    except Exception as e:
                        st.error(f"❌ Login failed: {e}")
                else:
                    st.error('Please enter both email and password')
        
        elif choice == 'Forget Password':
            st.subheader("Reset Password")
            email = st.text_input('Enter your email address')
            if st.button('Send Reset Link'):
                if email:
                    success, message = reset_password(email)
                    if success:
                        st.success("Password reset email sent successfully.")
                    else:
                        st.warning(f"Password reset failed: {message}")
                else:
                    st.error('Please enter your email address')
    
    if st.session_state.signout:
        st.success(f"✅ Welcome, {st.session_state.username}!")
        st.text(f'Email: {st.session_state.useremail}')
        if st.button('Sign out'):
            logout_user()
            st.rerun()

def logout_user():
    """Logout user and clear session state"""
    st.session_state.signout = False
    st.session_state.signedout = False
    st.session_state.username = ''
    st.session_state.useremail = ''
    st.session_state.admin_authenticated = False
    st.session_state.admin_username = None

def initialize_session_state():
    """Initialize session state variables"""
    if 'ai_models' not in st.session_state:
        st.session_state.ai_models = AI_MODELS.copy()
    
    if 'model_usage_history' not in st.session_state:
        st.session_state.model_usage_history = []
    
    if 'user_posts' not in st.session_state:
        st.session_state.user_posts = []
    
    if 'oauth_connected' not in st.session_state:
        st.session_state.oauth_connected = {
            'twitter': False,
            'linkedin': False
        }
    
    # Initialize admin authentication
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if 'admin_username' not in st.session_state:
        st.session_state.admin_username = None

def create_agent(agent_type, custom_instructions=None):
    """Create an AI agent based on type"""
    base_instructions = {
        "caption_agent": [
            "Create short, catchy social media captions (under 25 words).",
            "Focus on aesthetic and natural elements.",
            "Include relevant hashtags for product, materials, and style.",
            "Always end with #Herbally and #HerballyShop.",
            "Maintain a serene, wellness-focused tone.",
            "Structure the output as a single line of text."
        ],
        "image_prompt_agent": [
            "Create detailed image generation prompts for product photography.",
            "Focus on serene, wellness-focused scenes.",
            "Include specific details about lighting, setting, and mood.",
            "Emphasize natural elements and botanical features.",
            "Describe composition and artistic style.",
            "Specify desired atmosphere and emotional impact.",
            "Include technical details for high-quality image generation."
        ],
        "email_agent": [
            "Create professional, engaging emails.",
            "Focus on sustainable fashion and eco-friendly practices.",
            "Format subject line as: 'Subject: [subject line]'",
            "Subject line should be concise and attention-grabbing.",
            "Maintain a warm, personal tone in the body.",
            "Include a call to action.",
            "End with a professional signature.",
            "Format as: Subject: [subject line]\n\n[email body]"
        ]
    }
    
    instructions = base_instructions.get(agent_type, [])
    if custom_instructions:
        instructions.extend(custom_instructions)
    
    return Agent(
        name=f"{agent_type.replace('_', ' ').title()}",
        description=f"Custom {agent_type.replace('_', ' ')} agent",
        model=Gemini(id="gemini-2.0-flash", api_key=os.getenv("TEXT_API_KEY")),
        markdown=True,
        debug_mode=True,
        instructions=instructions
    )

def generate_content_with_agent(agent_type, prompt, genre, content_type, temperature=0.8):
    """Generate content using specified agent with custom parameters"""
    try:
        agent = create_agent(agent_type)
        
        # Create enhanced prompt with genre and content type
        enhanced_prompt = f"""
        Generate {content_type} content in {genre} style for the following prompt:
        {prompt}
        
        Requirements:
        - Style: {genre}
        - Content Type: {content_type}
        - Creativity Level: High (temperature: {temperature})
        - Platform-specific optimization
        """
        
        response = agent.run(enhanced_prompt)
        
        # Extract content from RunResponse
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Update usage statistics
        if agent_type in st.session_state.ai_models:
            st.session_state.ai_models[agent_type]["usage_count"] += 1
            st.session_state.model_usage_history.append({
                "timestamp": datetime.now(),
                "agent": agent_type,
                "prompt": prompt,
                "genre": genre,
                "content_type": content_type
            })
        
        return content
    except Exception as e:
        st.error(f"Error generating content: {str(e)}")
        return None

def generate_image_with_dalle3(image_prompt):
    """Generate image using DALL-E 3"""
    try:
        # Generate image using DALL-E 3
        result = azure_client.images.generate(
            model='dall-e-3',
            prompt=image_prompt,
            n=1,
            size="1024x1024"
        )
        
        # Retrieve the generated image
        image_url = result.data[0].url
        response = httpx.get(image_url)
        
        if response.status_code == 200:
            # Convert to PIL Image
            image = Image.open(BytesIO(response.content))
            return image
        else:
            st.error(f"Failed to download image from DALL-E 3: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error generating image with DALL-E 3: {str(e)}")
        return None

def generate_actual_image(image_prompt, model="gemini"):
    """Generate actual image from prompt using selected model"""
    try:
        if model == "gemini":
            # Use Google's Gemini for image generation
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=image_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            # Extract image from response
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    return image
            
            # Fallback: If no image generated
            st.warning("Image generation not available, using placeholder")
            return None
            
        elif model == "dalle":
            return generate_image_with_dalle3(image_prompt)
        
        else:
            st.error(f"Unknown model: {model}")
            return None
        
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def generate_image_with_agent(prompt, genre, content_type, temperature=0.8):
    """Generate image using image prompt agent and create actual image"""
    try:
        import re
        
        # Detect herb mentions in the prompt
        used_herbs = [herb for herb in COLOR_PALETTE if re.search(rf"\b{herb}\b", prompt, re.IGNORECASE)]
        
        # Build dye description
        dye_descriptions = [
            f"{herb.capitalize()}-infused fabric in HEX {COLOR_PALETTE[herb]}"
            for herb in used_herbs
        ]
        dye_text = ", ".join(dye_descriptions) if dye_descriptions else "herbal-dyed fabrics from the brand's palette"
        
        # Create concise, high-quality image prompt
        enhanced_prompt = f"Professional 4K photorealistic image of a person wearing luxurious {dye_text} in {genre} style, {content_type} format, with anatomically correct facial features including clear eyes, nose, and mouth, natural lighting, serene {genre} mood, botanical elements, high-end photography quality, {prompt}"
        
        # Generate actual image from the prompt
        generated_image = generate_actual_image(enhanced_prompt)
        
        # Update usage statistics
        if "image_prompt_agent" in st.session_state.ai_models:
            st.session_state.ai_models["image_prompt_agent"]["usage_count"] += 1
            st.session_state.model_usage_history.append({
                "timestamp": datetime.now(),
                "agent": "image_prompt_agent",
                "prompt": prompt,
                "genre": genre,
                "content_type": content_type
            })
        
        return enhanced_prompt, generated_image
    except Exception as e:
        st.error(f"Error generating image prompt: {str(e)}")
        return None, None

def generate_platform_specific_image(prompt, genre, content_type, platform, temperature=0.8, model="gemini"):
    """Generate platform-specific image with tailored settings"""
    try:
        import re
        
        # Detect herb mentions in the prompt
        used_herbs = [herb for herb in COLOR_PALETTE if re.search(rf"\b{herb}\b", prompt, re.IGNORECASE)]
        
        # Build dye description
        dye_descriptions = [
            f"{herb.capitalize()}-infused fabric in HEX {COLOR_PALETTE[herb]}"
            for herb in used_herbs
        ]
        dye_text = ", ".join(dye_descriptions) if dye_descriptions else "herbal-dyed fabrics from the brand's palette"
        
        # Platform-specific image settings
        platform_settings = {
            "Twitter": "casual lifestyle shot, trendy aesthetic, social media friendly, vibrant colors",
            "LinkedIn": "professional corporate setting, business attire, sophisticated lighting, premium quality",
            "Instagram": "visual aesthetic focus, lifestyle photography, artistic composition, high-end fashion"
        }
        
        platform_style = platform_settings.get(platform, "professional photography")
        
        # Create platform-specific image prompt
        enhanced_prompt = f"Professional 4K photorealistic image of a person wearing luxurious {dye_text} in {genre} style, {content_type} format for {platform}, with anatomically correct facial features including clear eyes, nose, and mouth, {platform_style}, natural lighting, serene {genre} mood, botanical elements, high-end photography quality, {prompt}"
        
        # Generate actual image from the prompt using selected model
        generated_image = generate_actual_image(enhanced_prompt, model)
        
        # Update usage statistics for image generation
        if "image_prompt_agent" in st.session_state.ai_models:
            st.session_state.ai_models["image_prompt_agent"]["usage_count"] += 1
            st.session_state.model_usage_history.append({
                "timestamp": datetime.now(),
                "agent": "image_prompt_agent",
                "prompt": prompt,
                "genre": genre,
                "content_type": content_type,
                "platform": platform,
                "model": model
            })
        
        return enhanced_prompt, generated_image
    except Exception as e:
        st.error(f"Error generating {platform} image: {str(e)}")
        return None, None

def generate_platform_specific_caption(prompt, genre, content_type, platform, temperature=0.8):
    """Generate platform-specific caption with tailored tone and style"""
    try:
        # Platform-specific caption instructions
        platform_instructions = {
            "Twitter": [
                "Create ONE short, catchy social media caption (under 280 characters)",
                "Focus on trendy and aesthetic elements",
                "Include trending and relevant hashtags",
                "Maintain a casual, engaging tone",
                "Use emojis appropriately",
                "Make it shareable and relatable",
                "Always end with #Herbally and #HerballyShop",
                "Generate ONLY ONE caption, not multiple options"
            ],
            "LinkedIn": [
                "Create ONE professional, business-focused caption (under 100 words)",
                "Focus on industry insights, sustainability, and business value",
                "Include relevant industry hashtags",
                "Maintain a formal, professional tone",
                "Highlight business benefits and industry impact",
                "Include a call to action for professional engagement",
                "Always end with #Herbally and #SustainableFashion",
                "Generate ONLY ONE caption, not multiple options"
            ],
            "Instagram": [
                "Create ONE engaging, visual-focused caption (under 100 words)",
                "Focus on aesthetic and lifestyle elements",
                "Include relevant hashtags for product, materials, and style",
                "Maintain a creative, artistic tone",
                "Emphasize visual appeal and lifestyle",
                "Include a call to action for engagement",
                "Always end with #Herbally and #HerballyShop",
                "Generate ONLY ONE caption, not multiple options"
            ]
        }
        
        instructions = platform_instructions.get(platform, platform_instructions["Twitter"])
        
        # Create platform-specific caption prompt with clear single output instruction
        caption_prompt = f"""
        Generate ONE {content_type} caption in {genre} style for {platform}:
        {prompt}
        
        Requirements:
        - Style: {genre}
        - Content Type: {content_type}
        - Platform: {platform}
        - Creativity Level: High (temperature: {temperature})
        - {chr(10).join(instructions)}
        
        IMPORTANT: Generate ONLY ONE caption. Do not provide multiple options or alternatives.
        """
        
        # Create agent with platform-specific instructions
        agent = Agent(
            name=f"{platform} Caption Agent",
            description=f"Generate ONE {platform}-specific caption",
            model=Gemini(id="gemini-2.0-flash", api_key=os.getenv("TEXT_API_KEY")),
            markdown=True,
            debug_mode=True,
            instructions=instructions
        )
        
        response = agent.run(caption_prompt)
        caption = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up the caption to ensure it's a single caption
        if caption:
            # Remove any markdown formatting that might indicate multiple options
            caption = caption.replace('**Option 1:**', '').replace('**Option 2:**', '').replace('**Option 3:**', '')
            caption = caption.replace('**Option 1 (', '').replace('**Option 2 (', '').replace('**Option 3 (', '')
            caption = caption.replace('**Focus on', '').replace('):**', '')
            
            # Take only the first paragraph if multiple are generated
            lines = caption.strip().split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('**') and not line.startswith('>') and not line.startswith('Option'):
                    clean_lines.append(line)
            
            # Join the lines and take the first complete caption
            caption = ' '.join(clean_lines)
            
            # If caption is too long, take only the first sentence or reasonable length
            if len(caption) > 500:  # If too long, likely multiple captions
                sentences = caption.split('.')
                caption = '. '.join(sentences[:2]) + '.'  # Take first two sentences
        
        # Update usage statistics for caption generation
        if "caption_agent" in st.session_state.ai_models:
            st.session_state.ai_models["caption_agent"]["usage_count"] += 1
            st.session_state.model_usage_history.append({
                "timestamp": datetime.now(),
                "agent": "caption_agent",
                "prompt": prompt,
                "genre": genre,
                "content_type": content_type,
                "platform": platform
            })
        
        return caption.strip() if caption else None
    except Exception as e:
        st.error(f"Error generating {platform} caption: {str(e)}")
        return None

def generate_combined_content(prompt, genre, content_type, platforms, temperature=0.8, model="gemini"):
    """Generate both image prompt and caption together for multiple platforms"""
    try:
        platform_content = {}
        
        for platform in platforms:
            # Generate platform-specific caption
            caption = generate_platform_specific_caption(prompt, genre, content_type, platform, temperature)
            
            # Generate platform-specific image
            image_prompt, generated_image = generate_platform_specific_image(prompt, genre, content_type, platform, temperature, model)
            
            platform_content[platform] = {
                "caption": caption,
                "image_prompt": image_prompt,
                "image": generated_image
            }
        
        return platform_content
    except Exception as e:
        st.error(f"Error generating combined content: {str(e)}")
        return None

def admin_dashboard():
    """Admin Panel Dashboard"""
    st.header("🔧 Admin Dashboard")
    
    # Model Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_models = len(st.session_state.ai_models)
        active_models = sum(1 for model in st.session_state.ai_models.values() if model["status"] == "active")
        st.metric("Total Models", total_models)
        st.metric("Active Models", active_models)
    
    with col2:
        total_usage = sum(model["usage_count"] for model in st.session_state.ai_models.values())
        st.metric("Total Usage", total_usage)
    
    with col3:
        avg_accuracy = sum(model["accuracy"] for model in st.session_state.ai_models.values()) / len(st.session_state.ai_models)
        st.metric("Avg Accuracy", f"{avg_accuracy:.2%}")
    
    # Model Management
    st.subheader("🤖 AI Model Management")
    
    # Create a DataFrame for better display
    model_data = []
    for key, model in st.session_state.ai_models.items():
        model_data.append({
            "Model Name": model["name"],
            "Status": model["status"],
            "Usage Count": model["usage_count"],
            "Accuracy": f"{model['accuracy']:.2%}",
            "Engagement Rate": f"{model['engagement_rate']:.2%}",
            "Model Type": model["model"]
        })
    
    df = pd.DataFrame(model_data)
    st.dataframe(df, use_container_width=True)
    
    # Model Controls
    st.subheader("Model Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Activate/Deactivate Models**")
        for key, model in st.session_state.ai_models.items():
            current_status = model["status"]
            new_status = st.selectbox(
                f"{model['name']}",
                ["active", "inactive", "maintenance"],
                index=["active", "inactive", "maintenance"].index(current_status),
                key=f"status_{key}"
            )
            if new_status != current_status:
                st.session_state.ai_models[key]["status"] = new_status
                st.success(f"Updated {model['name']} status to {new_status}")
    
    with col2:
        st.write("**Model Performance Metrics**")
        
        # Accuracy Chart
        accuracy_data = {
            "Model": [model["name"] for model in st.session_state.ai_models.values()],
            "Accuracy": [model["accuracy"] for model in st.session_state.ai_models.values()]
        }
        accuracy_df = pd.DataFrame(accuracy_data)
        
        fig = px.bar(accuracy_df, x="Model", y="Accuracy", 
                    title="Model Accuracy Comparison",
                    color="Accuracy", color_continuous_scale="viridis")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Usage Analytics
    st.subheader("📊 Usage Analytics")
    
    if st.session_state.model_usage_history:
        usage_df = pd.DataFrame(st.session_state.model_usage_history)
        usage_df['date'] = pd.to_datetime(usage_df['timestamp']).dt.date
        
        # Daily usage chart
        daily_usage = usage_df.groupby(['date', 'agent']).size().reset_index(name='count')
        
        fig = px.line(daily_usage, x='date', y='count', color='agent',
                     title="Daily Model Usage",
                     labels={'count': 'Usage Count', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Genre distribution
        genre_dist = usage_df['genre'].value_counts()
        fig = px.pie(values=genre_dist.values, names=genre_dist.index,
                    title="Content Genre Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No usage data available yet.")

def cleanup_files(image_path):
    """Clean up temporary image files"""
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
    except Exception as e:
        st.warning(f"Could not clean up file {image_path}: {str(e)}")

def save_image_for_posting(image, platform, model):
    """Save PIL image to temporary file for posting"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "tmp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time())
        filename = f"{platform.lower()}_{model}_{timestamp}.png"
        filepath = os.path.join(temp_dir, filename)
        
        # Save image
        image.save(filepath, "PNG")
        return filepath
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

def post_to_twitter(image_path, caption):
    """Post to Twitter using the exact same method as twitter.py"""
    try:
        print(f"DEBUG: Starting Twitter post with image: {image_path}")
        print(f"DEBUG: Caption length: {len(caption)}")
        
        # Check if API keys are available
        if not all([TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET]):
            print("DEBUG: Twitter API keys missing")
            return False, "Twitter API keys are missing. Please check your environment variables."
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"DEBUG: Image file not found: {image_path}")
            return False, f"Image file not found: {image_path}"
        
        print("DEBUG: Creating Twitter API instances...")
        # Create API instances exactly like twitter.py
        auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
        auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
        
        api = tweepy.API(auth)
        newapi = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_SECRET,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET,
        )
        
        print("DEBUG: Uploading media to Twitter...")
        # Upload media using API v1.1 (exactly like twitter.py)
        media = api.media_upload(image_path)
        print(f"DEBUG: Media uploaded, ID: {media.media_id}")
        
        print("DEBUG: Creating tweet...")
        # Create tweet using API v2 (exactly like twitter.py)
        post_result = newapi.create_tweet(text=caption, media_ids=[media.media_id])
        print(f"DEBUG: Tweet result: {post_result}")
        
        if post_result and post_result.data:
            print(f"DEBUG: Twitter post successful, ID: {post_result.data['id']}")
            return True, f"Successfully posted to Twitter! Tweet ID: {post_result.data['id']}"
        else:
            print("DEBUG: Twitter post failed - no response data")
            return False, "Failed to post to Twitter: No response data"
    except Exception as e:
        print(f"DEBUG: Twitter post error: {str(e)}")
        return False, f"Failed to post to Twitter: {str(e)}"

def post_to_linkedin(image_path, caption):
    """Post to LinkedIn using the exact same method as linkedin.py"""
    try:
        print(f"DEBUG: Starting LinkedIn post with image: {image_path}")
        print(f"DEBUG: Caption length: {len(caption)}")
        
        # Check if LinkedIn credentials are available
        if not all([LINKEDIN_ACCESS_TOKEN, LINKEDIN_ORG_ID]):
            print("DEBUG: LinkedIn credentials missing")
            return False, "LinkedIn credentials are missing. Please check your environment variables."
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"DEBUG: Image file not found: {image_path}")
            return False, f"Image file not found: {image_path}"
        
        print("DEBUG: Creating LinkedIn headers...")
        # Create headers exactly like linkedin.py
        headers = {
            'Authorization': f'Bearer {LINKEDIN_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        }
        
        print("DEBUG: Registering LinkedIn image upload...")
        # Register image upload (exactly like linkedin.py)
        url = 'https://api.linkedin.com/v2/assets?action=registerUpload'
        body = {
            "registerUploadRequest": {
                "recipes": [
                    "urn:li:digitalmediaRecipe:feedshare-image"
                ],
                "owner": LINKEDIN_ORG_URN,
                "serviceRelationships": [
                    {
                        "relationshipType": "OWNER",
                        "identifier": "urn:li:userGeneratedContent"
                    }
                ]
            }
        }
        response = requests.post(url, headers=headers, data=json.dumps(body))
        print(f"DEBUG: LinkedIn registration response: {response.status_code}")
        
        if response.status_code != 200:
            print(f"DEBUG: LinkedIn registration failed: {response.text}")
            return False, f"Failed to register LinkedIn upload: {response.status_code} {response.text}"
        
        data = response.json()
        upload_url = data['value']['uploadMechanism']['com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest']['uploadUrl']
        asset_urn = data['value']['asset']
        print(f"DEBUG: LinkedIn upload URL and asset URN obtained")
        
        print("DEBUG: Uploading image to LinkedIn...")
        # Upload image (exactly like linkedin.py)
        with open(image_path, 'rb') as f:
            image_data = f.read()
        upload_headers = {
            'Authorization': f'Bearer {LINKEDIN_ACCESS_TOKEN}',
            'Content-Type': 'application/octet-stream'
        }
        response = requests.put(upload_url, data=image_data, headers=upload_headers)
        print(f"DEBUG: LinkedIn image upload response: {response.status_code}")
        
        if response.status_code not in [200, 201, 202]:
            print(f"DEBUG: LinkedIn image upload failed: {response.text}")
            return False, f"LinkedIn image upload failed: {response.status_code} {response.text}"

        print("DEBUG: Creating LinkedIn post...")
        # Create post (exactly like linkedin.py)
        url = 'https://api.linkedin.com/v2/ugcPosts'
        post_body = {
            "author": LINKEDIN_ORG_URN,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": caption
                    },
                    "shareMediaCategory": "IMAGE",
                    "media": [
                        {
                            "status": "READY",
                            "description": {
                                "text": "Product showcase"
                            },
                            "media": asset_urn,
                            "title": {
                                "text": "Product Image"
                            }
                        }
                    ]
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }
        response = requests.post(url, headers=headers, data=json.dumps(post_body))
        print(f"DEBUG: LinkedIn post response: {response.status_code}")
        
        if response.status_code == 201:
            print("DEBUG: LinkedIn post successful")
            return True, "Successfully posted to LinkedIn!"
        else:
            print(f"DEBUG: LinkedIn post failed: {response.text}")
            return False, f"Failed to create LinkedIn post: {response.status_code} {response.text}"
    except Exception as e:
        print(f"DEBUG: LinkedIn post error: {str(e)}")
        return False, f"Failed to post to LinkedIn: {str(e)}"

def post_content_to_platform(platform, caption, image, image_model):
    """Post content to the specified platform"""
    try:
        print(f"DEBUG: post_content_to_platform called for {platform}")
        print(f"DEBUG: Caption: {caption[:100]}...")
        print(f"DEBUG: Image type: {type(image)}")
        print(f"DEBUG: Image model: {image_model}")
        
        # Check inputs
        if not caption:
            print("DEBUG: No caption provided")
            return False, f"No caption provided for {platform}"
        
        if not image:
            print("DEBUG: No image provided")
            return False, f"No image provided for {platform}"
        
        print("DEBUG: Saving image to temporary file...")
        # Save image to temporary file
        image_path = save_image_for_posting(image, platform, image_model)
        if not image_path:
            print("DEBUG: Failed to save image")
            return False, f"Failed to save image for {platform}"
        
        print(f"DEBUG: Image saved to: {image_path}")
        
        # Post to platform
        if platform.lower() == "twitter":
            print("DEBUG: Calling post_to_twitter...")
            success, message = post_to_twitter(image_path, caption)
        elif platform.lower() == "linkedin":
            print("DEBUG: Calling post_to_linkedin...")
            success, message = post_to_linkedin(image_path, caption)
        else:
            print(f"DEBUG: Unsupported platform: {platform}")
            return False, f"Unsupported platform: {platform}"
        
        print(f"DEBUG: Platform posting result - Success: {success}, Message: {message}")
        
        # Clean up temporary file
        if success:
            print("DEBUG: Cleaning up temporary file...")
            cleanup_files(image_path)
        
        return success, message
        
    except Exception as e:
        print(f"DEBUG: Error in post_content_to_platform: {str(e)}")
        return False, f"Error posting to {platform}: {str(e)}"

def check_environment_variables():
    """Check and display the status of environment variables"""
    st.subheader("🔧 API Configuration Status")
    
    # Twitter variables
    twitter_vars = {
        "CONSUMER_KEY": os.getenv("CONSUMER_KEY"),
        "CONSUMER_SECRET": os.getenv("CONSUMER_SECRET"),
        "ACCESS_KEY": os.getenv("ACCESS_KEY"),
        "ACCESS_SECRET": os.getenv("ACCESS_SECRET"),
        "BEARER_TOKEN": os.getenv("BEARER_TOKEN")
    }
    
    # LinkedIn variables
    linkedin_vars = {
        "LINKEDIN_ACCESS_TOKEN": os.getenv("LINKEDIN_ACCESS_TOKEN"),
        "LINKEDIN_ORG_ID": os.getenv("LINKEDIN_ORG_ID")
    }
    
    # AI Model variables
    ai_vars = {
        "TEXT_API_KEY": os.getenv("TEXT_API_KEY"),
        "IMAGE_API_KEY": os.getenv("IMAGE_API_KEY"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT")
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Twitter API Variables:**")
        for var_name, var_value in twitter_vars.items():
            if var_value:
                st.success(f"✅ {var_name}: Set")
            else:
                st.error(f"❌ {var_name}: Missing")
    
    with col2:
        st.write("**LinkedIn API Variables:**")
        for var_name, var_value in linkedin_vars.items():
            if var_value:
                st.success(f"✅ {var_name}: Set")
            else:
                st.error(f"❌ {var_name}: Missing")
    
    with col3:
        st.write("**AI Model Variables:**")
        for var_name, var_value in ai_vars.items():
            if var_value:
                st.success(f"✅ {var_name}: Set")
            else:
                st.warning(f"⚠️ {var_name}: Missing")
    
    # Check if all required variables are set
    all_twitter_set = all(twitter_vars.values())
    all_linkedin_set = all(linkedin_vars.values())
    all_ai_set = all(ai_vars.values())
    
    st.markdown("---")
    
    if all_twitter_set:
        st.success("✅ All Twitter API variables are configured")
    else:
        st.error("❌ Some Twitter API variables are missing")
    
    if all_linkedin_set:
        st.success("✅ All LinkedIn API variables are configured")
    else:
        st.error("❌ Some LinkedIn API variables are missing")
    
    if all_ai_set:
        st.success("✅ All AI Model variables are configured")
    else:
        st.warning("⚠️ Some AI Model variables are missing - image generation may not work")
    
    st.info("💡 **Note:** Make sure your .env file is in the same directory as admin.py")

def user_dashboard():
    """User Panel Dashboard"""
    st.header("👤 User Dashboard")
    
    st.subheader("🔗 Platform Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Twitter (X) Integration**")
        if st.session_state.oauth_connected['twitter']:
            st.success("✅ Connected to Twitter")
            if st.button("Disconnect Twitter"):
                st.session_state.oauth_connected['twitter'] = False
                st.success("Disconnected from Twitter")
        else:
            st.warning("❌ Not connected to Twitter")
            if st.button("Connect Twitter"):
                # Simulate OAuth connection
                st.session_state.oauth_connected['twitter'] = True
                st.success("Connected to Twitter!")
    
    with col2:
        st.write("**LinkedIn Integration**")
        if st.session_state.oauth_connected['linkedin']:
            st.success("✅ Connected to LinkedIn")
            if st.button("Disconnect LinkedIn"):
                st.session_state.oauth_connected['linkedin'] = False
                st.success("Disconnected from LinkedIn")
        else:
            st.warning("❌ Not connected to LinkedIn")
            if st.button("Connect LinkedIn"):
                # Simulate OAuth connection
                st.session_state.oauth_connected['linkedin'] = True
                st.success("Connected to LinkedIn!")
    
    # Add environment variables check
    if st.button("🔧 Check API Configuration"):
        check_environment_variables()
    
    # Content Creation Interface
    st.subheader("🎨 Content Creation")
    
    # User input section
    col1, col2 = st.columns(2)
    
    with col1:
        prompt = st.text_area(
            "Enter your product description or prompt",
            placeholder="Example: A red sandalwood-hued cord set placed near indoor plants in filtered daylight.",
            height=100
        )
        
        genre = st.selectbox("Select Genre", CONTENT_GENRES)
        
        content_type = st.selectbox("Select Content Type", CONTENT_TYPES)
        
        image_model = st.selectbox(
            "Select Image Generation Model",
            ["gemini", "dalle"],
            format_func=lambda x: "Gemini" if x == "gemini" else "DALL-E 3",
            help="Choose between Gemini and DALL-E 3 for image generation"
        )
    
    with col2:
        temperature = st.slider("Creativity Level (Temperature)", 0.1, 1.0, 0.8, 0.1,
                              help="Higher values = more creative, Lower values = more focused")
        
        platforms = st.multiselect(
            "Select Platforms",
            ["Twitter", "LinkedIn", "Instagram"],
            default=["Twitter", "LinkedIn"]
        )
        
        # Content generation type
        generation_type = st.radio(
            "Select Generation Type",
            ["Combined (Image + Caption)", "Caption Only", "Image Only"],
            help="Combined generates both image and caption together for each platform"
        )
    
    # Generate content button
    if st.button("🚀 Generate Content", type="primary"):
        if not prompt:
            st.error("Please enter a product description")
        elif not platforms:
            st.error("Please select at least one platform")
        else:
            with st.spinner(f"Generating content for {len(platforms)} platform(s) using {image_model.upper()}..."):
                if generation_type == "Combined (Image + Caption)":
                    platform_content = generate_combined_content(
                        prompt, genre, content_type, platforms, temperature, image_model
                    )
                    
                    if platform_content:
                        st.success(f"Content generated successfully for {len(platforms)} platform(s)!")
                        
                        # Display generated content for each platform
                        st.subheader("Generated Content by Platform")
                        
                        for platform in platforms:
                            if platform in platform_content:
                                content = platform_content[platform]
                                
                                with st.expander(f"📱 {platform} Content", expanded=True):
                                    # Create two columns for caption and image
                                    caption_col, image_col = st.columns(2)
                                    
                                    with caption_col:
                                        st.markdown(f"### 📝 {platform} Caption")
                                        st.text_area(f"{platform} Caption", content["caption"], height=150, key=f"caption_{platform}")
                                        
                                        # Character count for Twitter
                                        if platform == "Twitter":
                                            char_count = len(content["caption"])
                                            if char_count > 280:
                                                st.warning(f"⚠️ Caption is {char_count - 280} characters over Twitter's 280 limit")
                                            else:
                                                st.success(f"✅ Character count: {char_count}/280")
                                    
                                    with image_col:
                                        st.markdown(f"### 🖼️ {platform} Image ({image_model.upper()})")
                                        if content["image"]:
                                            st.image(content["image"], caption=f"{platform} Generated Image", use_column_width=True)
                                            
                                            # Add download button for the image
                                            img_buffer = BytesIO()
                                            content["image"].save(img_buffer, format='PNG')
                                            img_buffer.seek(0)
                                            st.download_button(
                                                label=f"📥 Download {platform} Image",
                                                data=img_buffer.getvalue(),
                                                file_name=f"{platform.lower()}_{image_model}_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png"
                                            )
                                        else:
                                            st.info("Image generation not available")
                                    
                                    # Platform-specific preview
                                    st.markdown(f"### {platform} Style Guide")
                                    if platform == "Twitter":
                                        st.markdown("- **Tone:** Casual and engaging")
                                        st.markdown("- **Focus:** Trendy and aesthetic")
                                        st.markdown("- **Hashtags:** #Herbally and #HerballyShop")
                                    elif platform == "LinkedIn":
                                        st.markdown("- **Tone:** Professional and business-focused")
                                        st.markdown("- **Focus:** Industry insights and sustainability")
                                        st.markdown("- **Hashtags:** #Herbally and #SustainableFashion")
                                    elif platform == "Instagram":
                                        st.markdown("- **Tone:** Visual and aesthetic focus")
                                        st.markdown("- **Focus:** Lifestyle and social media friendly")
                                        st.markdown("- **Hashtags:** #Herbally and #HerballyShop")
                        
                        # Save to user posts
                        for platform in platforms:
                            if platform in platform_content:
                                content = platform_content[platform]
                                st.session_state.user_posts.append({
                                    "timestamp": datetime.now(),
                                    "prompt": prompt,
                                    "genre": genre,
                                    "content_type": content_type,
                                    "platforms": [platform],
                                    "generation_type": generation_type,
                                    "caption": content["caption"],
                                    "image_prompt": content["image_prompt"],
                                    "has_image": content["image"] is not None,
                                    "image_model": image_model,
                                    "temperature": temperature
                                })
                        
                        # Store platform content in session state for posting
                        st.session_state.platform_content = platform_content
                        st.session_state.platforms_to_post = platforms
                        st.session_state.image_model = image_model
                        
                        # Show platform status
                        connected_platforms = [p for p in platforms if st.session_state.oauth_connected.get(p.lower(), False)]
                        disconnected_platforms = [p for p in platforms if not st.session_state.oauth_connected.get(p.lower(), False)]
                        
                        if connected_platforms:
                            st.success(f"✅ Connected platforms: {', '.join(connected_platforms)}")
                        
                        if disconnected_platforms:
                            st.warning(f"❌ Disconnected platforms: {', '.join(disconnected_platforms)}")
                        
                        # Initialize posting results
                        if 'posting_results' not in st.session_state:
                            st.session_state.posting_results = {}
                        
                        # Manual posting button
                        st.markdown("---")
                        st.subheader("📤 Manual Posting")
                        
                        if st.button("🚀 Post to Selected Platforms", type="primary", key="manual_post_button"):
                            if not connected_platforms:
                                st.error("❌ No platforms are connected. Please connect to at least one platform first.")
                            else:
                                st.info(f"🚀 Posting to {len(connected_platforms)} platform(s)...")
                                
                                # Initialize posting results
                                st.session_state.posting_results = {}
                                
                                # Post to each connected platform
                                for platform in connected_platforms:
                                    if platform in platform_content:
                                        content = platform_content[platform]
                                        
                                        with st.spinner(f"📤 Posting to {platform}..."):
                                            success, message = post_content_to_platform(
                                                platform, 
                                                content["caption"], 
                                                content["image"], 
                                                image_model
                                            )
                                            
                                            # Store result
                                            st.session_state.posting_results[platform] = {
                                                "success": success, 
                                                "message": message
                                            }
                                            
                                            # Show immediate feedback
                                            if success:
                                                st.success(f"✅ {platform}: Posted successfully!")
                                            else:
                                                st.error(f"❌ {platform}: {message}")
                                
                                # Show final summary
                                st.markdown("---")
                                st.subheader("📊 Posting Summary")
                                
                                successful_posts = [p for p, r in st.session_state.posting_results.items() if r["success"]]
                                failed_posts = [p for p, r in st.session_state.posting_results.items() if not r["success"]]
                                
                                if successful_posts:
                                    st.success(f"🎉 **Successfully posted to {len(successful_posts)} platform(s):** {', '.join(successful_posts)}")
                                
                                if failed_posts:
                                    st.error(f"❌ **Failed to post to {len(failed_posts)} platform(s):** {', '.join(failed_posts)}")
                                
                                # Show detailed results
                                st.subheader("📋 Detailed Results")
                                for platform, result in st.session_state.posting_results.items():
                                    with st.expander(f"{platform} - {'✅ Success' if result['success'] else '❌ Failed'}", expanded=True):
                                        if result["success"]:
                                            st.success(f"**Message:** {result['message']}")
                                        else:
                                            st.error(f"**Error:** {result['message']}")
                                            
                                            # Add troubleshooting tips for common errors
                                            if "API keys" in result['message'].lower() or "credentials" in result['message'].lower():
                                                st.info("💡 **Troubleshooting:** Check your API credentials in the environment variables.")
                                            elif "not connected" in result['message'].lower():
                                                st.info("💡 **Troubleshooting:** Make sure you're connected to the platform first.")
                                            elif "image" in result['message'].lower():
                                                st.info("💡 **Troubleshooting:** There might be an issue with the image format or size.")
                
                elif generation_type == "Caption Only":
                    st.subheader("Generated Captions by Platform")
                    
                    for platform in platforms:
                        caption = generate_platform_specific_caption(prompt, genre, content_type, platform, temperature)
                        
                        if caption:
                            with st.expander(f"📝 {platform} Caption", expanded=True):
                                st.text_area(f"{platform} Caption", caption, height=150, key=f"caption_only_{platform}")
                                
                                # Character count for Twitter
                                if platform == "Twitter":
                                    char_count = len(caption)
                                    if char_count > 280:
                                        st.warning(f"⚠️ Caption is {char_count - 280} characters over Twitter's 280 limit")
                                    else:
                                        st.success(f"✅ Character count: {char_count}/280")
                            
                            # Save to user posts
                            st.session_state.user_posts.append({
                                "timestamp": datetime.now(),
                                "prompt": prompt,
                                "genre": genre,
                                "content_type": content_type,
                                "platforms": [platform],
                                "generation_type": generation_type,
                                "caption": caption,
                                "image_prompt": None,
                                "has_image": False,
                                "image_model": None,
                                "temperature": temperature
                            })
                
                elif generation_type == "Image Only":
                    st.subheader(f"Generated Images by Platform ({image_model.upper()})")
                    
                    for platform in platforms:
                        image_prompt, generated_image = generate_platform_specific_image(prompt, genre, content_type, platform, temperature, image_model)
                        
                        if image_prompt:
                            with st.expander(f"🖼️ {platform} Image", expanded=True):
                                if generated_image:
                                    st.image(generated_image, caption=f"{platform} Generated Image", use_column_width=True)
                                    
                                    # Add download button for the image
                                    img_buffer = BytesIO()
                                    generated_image.save(img_buffer, format='PNG')
                                    img_buffer.seek(0)
                                    st.download_button(
                                        label=f"📥 Download {platform} Image",
                                        data=img_buffer.getvalue(),
                                        file_name=f"{platform.lower()}_{image_model}_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                        mime="image/png"
                                    )
                                else:
                                    st.info("Image generation not available")
                            
                            # Save to user posts
                            st.session_state.user_posts.append({
                                "timestamp": datetime.now(),
                                "prompt": prompt,
                                "genre": genre,
                                "content_type": content_type,
                                "platforms": [platform],
                                "generation_type": generation_type,
                                "caption": None,
                                "image_prompt": image_prompt,
                                "has_image": generated_image is not None,
                                "image_model": image_model,
                                "temperature": temperature
                            })
    
    # User's Post History
    if st.session_state.user_posts:
        st.subheader("📝 Your Post History")
        
        # Convert to DataFrame for display
        posts_data = []
        for post in st.session_state.user_posts:
            posts_data.append({
                "Date": post["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Genre": post["genre"],
                "Content Type": post["content_type"],
                "Platforms": ", ".join(post["platforms"]),
                "Generation Type": post["generation_type"],
                "Has Caption": "✅" if post.get("caption") else "❌",
                "Has Image": "✅" if post.get("has_image") else "❌",
                "Image Model": post.get("image_model", "N/A"),
                "Temperature": post["temperature"]
            })
        
        posts_df = pd.DataFrame(posts_data)
        st.dataframe(posts_df, use_container_width=True)
        
        # Download option
        csv = posts_df.to_csv(index=False)
        st.download_button(
            label="Download Post History",
            data=csv,
            file_name="user_posts_history.csv",
            mime="text/csv"
        )

def main():
    st.set_page_config(
        page_title="Social Media Agent - Admin Panel",
        page_icon="🔧",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🔧 Social Media Agent - Admin & User Panel")
    
    # Create tabs for Admin and User panels
    admin_tab, user_tab = st.tabs(["🔧 Admin Panel", "👤 User Panel"])
    
    with admin_tab:
        if not st.session_state.admin_authenticated:
            admin_login()
        else:
            st.markdown(f"### 👋 Welcome, {st.session_state.admin_username}!")
            if st.button("🚪 Logout", key="admin_logout_btn"):
                logout_user()
                st.success("Logged out successfully!")
                st.rerun()
            st.markdown("---")
            admin_dashboard()
    
    with user_tab:
        user_dashboard()

# Firebase Authentication Functions
def sign_up_with_email_and_password(email, password, username=None, return_secure_token=True):
    """Sign up user with Firebase Authentication"""
    try:
        rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": return_secure_token
        }
        if username:
            payload["displayName"] = username
        payload = json.dumps(payload)
        r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
        try:
            return r.json()['email']
        except:
            st.warning(r.json())
    except Exception as e:
        st.warning(f'Signup failed: {e}')

def sign_in_with_email_and_password(email=None, password=None, return_secure_token=True):
    """Sign in user with Firebase Authentication"""
    rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

    try:
        payload = {
            "returnSecureToken": return_secure_token
        }
        if email:
            payload["email"] = email
        if password:
            payload["password"] = password
        payload = json.dumps(payload)
        r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
        try:
            data = r.json()
            user_info = {
                'email': data['email'],
                'username': data.get('displayName')  # Retrieve username if available
            }
            return user_info
        except:
            st.warning(data)
    except Exception as e:
        st.warning(f'Signin failed: {e}')

def reset_password(email):
    """Reset password using Firebase Authentication"""
    try:
        rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode"
        payload = {
            "email": email,
            "requestType": "PASSWORD_RESET"
        }
        payload = json.dumps(payload)
        r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
        if r.status_code == 200:
            return True, "Reset email sent"
        else:
            error_message = r.json().get('error', {}).get('message')
            return False, error_message
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    main() 