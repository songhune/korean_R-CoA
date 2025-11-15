"""
Test API Keys
OpenAI와 Anthropic API 키를 테스트합니다.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

print("="*70)
print("API KEY TEST")
print("="*70)
print()

# Test OpenAI
print("[1/2] Testing OpenAI API Key...")
openai_key = os.environ.get('OPENAI_API_KEY', '')
if openai_key:
    print(f"  Key found: {openai_key[:20]}...{openai_key[-10:]}")
    print(f"  Key length: {len(openai_key)} characters")

    try:
        import openai
        client = openai.OpenAI(api_key=openai_key)

        # Simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )

        print("  ✅ OpenAI API Key is VALID!")
        print(f"  Response: {response.choices[0].message.content}")

    except openai.AuthenticationError as e:
        print("  ❌ OpenAI API Key is INVALID!")
        print(f"  Error: {e}")
        print()
        print("  해결방법:")
        print("  1. https://platform.openai.com/api-keys 에서 새 API 키 생성")
        print("  2. .env 파일의 OPENAI_API_KEY 업데이트")

    except Exception as e:
        print(f"  ⚠️  Error: {e}")
else:
    print("  ❌ OPENAI_API_KEY not found in .env")

print()

# Test Anthropic
print("[2/2] Testing Anthropic API Key...")
anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')
if anthropic_key:
    print(f"  Key found: {anthropic_key[:20]}...{anthropic_key[-10:]}")
    print(f"  Key length: {len(anthropic_key)} characters")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)

        # Simple test call
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )

        print("  ✅ Anthropic API Key is VALID!")
        print(f"  Response: {response.content[0].text}")

    except anthropic.AuthenticationError as e:
        print("  ❌ Anthropic API Key is INVALID!")
        print(f"  Error: {e}")
        print()
        print("  해결방법:")
        print("  1. https://console.anthropic.com/settings/keys 에서 새 API 키 생성")
        print("  2. .env 파일의 ANTHROPIC_API_KEY 업데이트")

    except Exception as e:
        print(f"  ⚠️  Error: {e}")
else:
    print("  ❌ ANTHROPIC_API_KEY not found in .env")

print()
print("="*70)
print("TEST COMPLETE")
print("="*70)
