import json
import os

from dotenv import load_dotenv
import requests
from langchain.tools import tool
import json

from helpers.db import DBHandler

load_dotenv(".env")

# db = DBHandler("test.db")

class SearchTools():

    @tool("Search the internet")
    def search_internet(query):
        """Useful to search the internet
        about a a given topic and return relevant results"""
        print("Searching the internet...")
        top_result_to_return = 5
        url = "https://google.serper.dev/search"

        payload = json.dumps(
            {"q": query, "num": top_result_to_return, "tbm": "nws"})
        # if db.get(json.dumps(payload)):
        #     return db.get(json.dumps(payload))
        # else:
        headers = {
            'X-API-KEY': os.environ['SERPER_API_KEY'],
            'content-type': 'application/json'
        }
        print("Searching for:", headers, payload)
        response = requests.request("POST", url, headers=headers, data=payload)

        # write respone to a file
        with open('search_results.json', 'w') as f:
            f.write(response.text)

        # check if there is an organic key
        if 'organic' not in response.json():
            return "Sorry, I couldn't find anything about that, there could be an error with you serper api key."
        else:
            results = response.json()['organic']
            # db.set(json.dumps(payload), results)
            return results
            # string = []
            # print("Results:", results[:top_result_to_return])
            # for result in results[:top_result_to_return]:
            #     try:
            #         # Attempt to extract the date
            #         date = result.get('date', 'Date not available')
            #         string.append('\n'.join([
            #             f"Title: {result['title']}",
            #             f"Link: {result['link']}",
            #             f"Date: {date}",  # Include the date in the output
            #             f"Snippet: {result['snippet']}",
            #             "\n-----------------"
            #         ]))
            #     except KeyError:
            #         next

            # return '\n'.join(string)

    def search_result_converter(results):
        string = []
        for result in results:
            try:
                # Attempt to extract the date
                date = result.get('date', 'Date not available')
                string.append('\n'.join([
                    f"Title: {result['title']}",
                    f"Link: {result['link']}",
                    f"Date: {date}",  # Include the date in the output
                    f"Snippet: {result['snippet']}",
                    "\n-----------------"
                ]))
            except KeyError:
                next

        return '\n'.join(string)