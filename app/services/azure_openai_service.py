import os

from openai import AzureOpenAI

from dotenv import load_dotenv


load_dotenv()


client = AzureOpenAI(

    api_key=os.getenv("AZURE_OPENAI_KEY"),

    api_version=os.getenv(
        "AZURE_OPENAI_API_VERSION"
    ),

    azure_endpoint=os.getenv(
        "AZURE_OPENAI_ENDPOINT"
    )
)


def ask_openai(question):

    try:

        response = client.chat.completions.create(

            model="gpt-35-turbo",

            messages=[

                {
                    "role": "system",
                    "content": "You are a retail AI assistant."
                },

                {
                    "role": "user",
                    "content": question
                }
            ],

            max_tokens=200
        )

        return response.choices[0].message.content


    except Exception as e:

        return str(e)