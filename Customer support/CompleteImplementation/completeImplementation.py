from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TextClassificationPipeline
import tensorflow as tf
import random
import speech_recognition as sr
import gtts
import playsound
import os

save_directory = 'Saved_model'

#loading the model 
tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(save_directory)
model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

#intent classes
intent_class_names= ['cancel_order', 'change_order', 'change_shipping_address',
                  'check_cancellation_fee', 'check_invoice', 'check_payment_methods',
                  'check_refund_policy', 'complaint', 'contact_customer_service',
                  'contact_human_agent', 'create_account', 'delete_account',
                  'delivery_options', 'delivery_period', 'edit_account',
                  'get_invoice', 'get_refund', 'newsletter_subscription',
                  'payment_issue', 'place_order', 'recover_password', 'registration_problems',
                  'review', 'set_up_shipping_address', 'switch_account', 'track_order', 'track_refund']

def classify_intent(text):
  test_text = text
  predict_input = tokenizer_fine_tuned.encode(
      test_text,
      truncation = True,
      padding = True,
      return_tensors = 'tf'
  )

  output = model_fine_tuned(predict_input)[0]

  prediction_value = tf.argmax(output, axis = 1).numpy()[0]

  predicted_class_name = intent_class_names[prediction_value]
  return predicted_class_name



def generate_response(intent: str, user_query: str = None) -> str:


    # Define response templates for each intent
    response_templates = {
        'cancel_order': [
            "I'm sorry to hear that you'd like to cancel your order. Could you please provide your order ID so we can proceed?",
            "To cancel your order, please provide the order number. I'll handle it right away."
        ],
        'change_order': [
            "To make changes to your order, please specify what changes you'd like to make along with your order ID.",
            "Could you provide your order ID and details on what you want to change? I'll update your order accordingly."
        ],
        'change_shipping_address': [
            "Please provide your order ID and the new shipping address, and I'll update it for you.",
            "I can help you change the shipping address. Could you provide the order ID and the new address details?"
        ],
        'check_cancellation_fee': [
            "To check if there's a cancellation fee, please provide your order ID.",
            "I can check the cancellation fee for you. Can you provide the order number?"
        ],
        'check_invoice': [
            "To check your invoice, please provide your order ID or account details.",
            "I can retrieve your invoice. Could you provide the necessary details, such as your order number?"
        ],
        'check_payment_methods': [
            "We accept several payment methods including credit/debit cards, PayPal, and bank transfers. Would you like more details?",
            "Our payment methods include credit cards, debit cards, PayPal, and more. Can I assist you with something specific?"
        ],
        'check_refund_policy': [
            "Our refund policy allows refunds within 30 days of purchase. Would you like more details on this?",
            "We offer refunds within 30 days. If you need more information, let me know."
        ],
        'complaint': [
            "I'm sorry to hear that you're not satisfied. Could you please provide more details about your issue?",
            "We apologize for any inconvenience. Can you tell me more about your complaint so I can assist you?"
        ],
        'contact_customer_service': [
            "You can reach our customer service via phone at 123-456-7890 or email at support@example.com.",
            "For customer service, please call 123-456-7890 or email support@example.com."
        ],
        'contact_human_agent': [
            "I will connect you with a human agent shortly. Please hold on.",
            "Connecting you to a human agent now. Please wait a moment."
        ],
        'create_account': [
            "To create an account, please provide your email address and desired password.",
            "I can help you create an account. Could you please provide your email and a preferred password?"
        ],
        'delete_account': [
            "To delete your account, please provide your account email and any other verification details.",
            "I'm sorry to see you go. Could you confirm your account details so we can proceed with the deletion?"
        ],
        'delivery_options': [
            "We offer several delivery options including standard, express, and overnight shipping. Which one would you like?",
            "Our delivery options range from standard to express. Would you like more details?"
        ],
        'delivery_period': [
            "The delivery period depends on your location and the chosen delivery option. Typically, it ranges from 3 to 7 business days.",
            "Delivery usually takes between 3 to 7 business days, depending on your location and selected shipping method."
        ],
        'edit_account': [
            "To edit your account details, please provide the changes you want to make along with your current account information.",
            "I can assist you with editing your account. Please specify what details you would like to update."
        ],
        'get_invoice': [
            "To get your invoice, please provide your order ID.",
            "I can fetch your invoice. Could you provide your order number or account details?"
        ],
        'get_refund': [
            "To process a refund, please provide your order ID and the reason for the refund.",
            "Refunds can be initiated with your order ID and some additional information. Could you provide those details?"
        ],
        'newsletter_subscription': [
            "To subscribe to our newsletter, please provide your email address.",
            "I can add you to our newsletter. Could you provide your email?"
        ],
        'payment_issue': [
            "I'm sorry to hear you're having payment issues. Could you provide more details or a specific error message?",
            "Let’s solve your payment issue. Could you please describe the problem or provide any error messages you received?"
        ],
        'place_order': [
            "To place an order, please provide the product details and your preferred payment method.",
            "I can help you place an order. Could you tell me what you'd like to purchase and your payment method?"
        ],
        'recover_password': [
            "To recover your password, please provide your registered email address.",
            "I can help with password recovery. Could you provide your account's email address?"
        ],
        'registration_problems': [
            "I'm sorry to hear you're having trouble registering. Could you describe the issue?",
            "Let’s resolve your registration issue. Can you provide more details?"
        ],
        'review': [
            "We'd love to hear your feedback! Could you provide your review or comments?",
            "Please share your feedback with us. We appreciate your input."
        ],
        'set_up_shipping_address': [
            "To set up a shipping address, please provide your full address details.",
            "I can set up your shipping address. Could you provide the complete address information?"
        ],
        'switch_account': [
            "To switch accounts, please log out of the current account and log in with the other account credentials.",
            "You can switch accounts by logging out and logging back in with a different account. Need further assistance?"
        ],
        'track_order': [
            "To track your order, please provide your order ID.",
            "I can track your order for you. Could you provide the order number?"
        ],
        'track_refund': [
            "To track your refund status, please provide your order ID and any related details.",
            "Refund tracking requires your order ID. Could you provide that information?"
        ],
    }

    # Get random response from the list of responses for the given intent
    if intent in response_templates:
        response = random.choice(response_templates[intent])
    else:
        response = "I'm not sure how to help with that. Could you please provide more details?"

    return response



def main():
    choice = input("Enter your choice (1 for text input or 2 for audio input): ")
    if choice == '1':
        # Text input
        text = input("Please type your input: ")
        intent = classify_intent(text)
        response = generate_response(intent, text)
        print(f'UserText :{text}')
        print(f'Intent : {intent}')
        print(f'Response : {response}')
        
    elif choice == '2':
        r = sr.Recognizer()
        with sr.Microphone() as src:
            print("Listening....")
            try:
                audio = r.listen(src)
                text = r.recognize_google(audio)
                intent = classify_intent(text)
                response = generate_response(intent, text)
                sound = gtts.gTTS(response, lang='en')
                sound.save('response.mp3')
                playsound.playsound("response.mp3")
                os.remove('response.mp3')
                
            
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

        print(f'UserText :{text}')
        print(f'Intent : {intent}')
        print(f'Response : {response}')
    
    else:
        print("Invalid choice. Please choose 1 or 2.")   


if __name__ == "__main__":
    main()  



