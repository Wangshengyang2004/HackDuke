from googletrans import Translator

def translate_chinese_to_english(chinese_text):
    translator = Translator()
    
    # Translate the Chinese text into English.
    translated = translator.translate(chinese_text, src='zh-CN', dest='en')
    
    # Return the translated text.
    return translated.text

# Example usage:
chinese_sentence = "你好，世界"  # Replace with your Chinese sentence.
translated_sentence = translate_chinese_to_english(chinese_sentence)
print("Chinese: ", chinese_sentence)
print("English: ", translated_sentence)