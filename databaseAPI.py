import boto3
import base64
import json
from airtable import Airtable
from flask import Flask

app = Flask(__name__)


class Worker:
    unique = 0
    isAuth = False
    userInfoTable = Airtable('appytOreQAh0kwjlt', "User Info", "key1B3UuMtQpUkWQS")
    designProjectTable = Airtable('appytOreQAh0kwjlt', "Design projects", "key1B3UuMtQpUkWQS")


@app.route("/")
def index():
    """Present some documentation"""
    return "<h2>Add API documentation here !!</h2>"


@app.route("/recommendation_from_ml/JournalText=<string:JournalText>&Title=<string:Title>")
def recommendation_from_ml(JournalText, Title):
    # ToDo - recommendation system based on text and title
    return {'Activities': ['Hiking', 'Basketball', 'Swimming', 'Workout'], 'Groups':
        ['Travel and Tourism', 'Sports and Recreation', 'Gym and Fitness'], 'Resources':
                ['World Health organization https://www.who.int/health-topics/mental-health#tab=tab_1',
                 'CDC https://www.cdc.gov/mentalhealth/index.htm',
                 'Psychological health https://www.mentalhealth.gov/basics/what-is-mental-health']}


def mood_from_ml(text, title):
    return 0


def upload_to_s3(filename):
    s3 = boto3.client('s3')
    s3.upload_file(filename, 'sigmoidhack', filename)


def make_attachment(url):
    return [{'url': url}]


@app.route(
    "/post_journal/username=<string:username>&userID=<int:userID>&date=<string:date>&title=<string:title>&journalText=<string:journalText>&journalImage=<string:journalImage>")
def post_journal(userID, username, date, title, journalText, journalImage):
    # Todo- get mood from backend
    if Worker.isAuth:
        filename = ""
        if journalImage:
            imgData = base64.b64decode(journalImage)
            filename = username + str(Worker.unique) + ".jpg"
            Worker.unique += 1
            with open(filename, 'wb') as f:
                f.write(imgData)
        mood = mood_from_ml(journalText, title)
        upload_to_s3(filename)
        url = "https://sigmoidhack.s3.amazonaws.com/" + filename
        Worker.designProjectTable.insert({"UserID": userID, "UserName": username, "Date": date, "Title": title,
                                          "JournalText": journalText, "JournalImage": make_attachment(url),
                                          "Mood": mood})


@app.route("/get_all_journals/username=<string:username>")
def get_all_journals(username):
    all_journals = []
    if Worker.isAuth:
        records = Worker.designProjectTable.search('UserName', str(username))
        for r in records:
            all_journals.append(r["fields"])
    return json.dumps(all_journals)


@app.route("/get_single_journal/<int:journalID>")
def get_single_journal(journalID):
    journal = {}
    if Worker.isAuth:
        fetched_journal = Worker.designProjectTable.search("JournalID", journalID)
        journal = fetched_journal[0]["fields"]
    return journal


@app.route("/post_signup/username=<string:username>&password=<string:password>")
def post_signup(username, password):
    if not Worker.userInfoTable.search("Username", str(username)):
        Worker.userInfoTable.insert({"Username": str(username), "Password": str(password), "Active": True})
        return {"Success": True}
    return {"Success": False}


@app.route("/post_login/username=<string:username>&password=<string:password>")
def post_login(username, password):
    if Worker.userInfoTable.search("Username", str(username)) and Worker.userInfoTable.search("Password",
                                                                                              str(password)):
        Worker.isAuth = True
    else:
        Worker.isAuth = False
    return {"Success": Worker.isAuth}


@app.route("/post_deleteAccount/username=<string:username>&password=<string:password>")
def post_deleteAccount(self, username, password):
    if self.userInfoTable.search("Username", str(username)) and self.userInfoTable.search("Password", str(password)):
        self.userInfoTable.delete_by_field("Username", username)
        # ToDO remove all journal entries in User Info
        return {"Success": True}
    return {"Success": False}


"""
Main Routine
"""
if __name__ == "__main__":
    app.run()
