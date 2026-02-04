import discord
import gpt_2_simple as gpt2
import os
import time
import requests
import sys
import tensorflow as tf
from random import randint
import re
import json


class MyClient(discord.Client):

    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))
        channel = client.get_channel(409102823272480769)

    async def on_message(self, message):
        recieved = message.content
        channel = message.channel
        if not message.author.bot:
            await parseCommands(client, channel, message)


async def parseCommands(client, channel, message):

    # get any message from this chat not from bot
    if channel == client.get_channel(409102823272480769):
        # RESPOND WITH THE GENERATED TEXT
        async with channel.typing():
            if message.content == "!new":
                await channel.send("Starting a new Conversation.")
                os.remove("conversation.txt")
            else:
                await channel.send(sendGeneratedResponse(message))


def sendGeneratedResponse(message):

    # GET CONVERSATION UP TO THIS POINT

    print("---------------------------------------------------\nAdding\n" + message.content + "\nto the context of the coversation")
    conversation_file = open('conversation.txt', 'a')
    conversation_file.write(message.content + "\n")
    conversation_file.close()

    new_conversation_file = open('conversation.txt', 'r')
    context = new_conversation_file.read()
    new_conversation_file.close()

    print("\n---------------------------------------------------\nConversation reads as:\n" + context + "\n---------------------------------------------------")


    response = ""
    # TRY THE PROCESSING
    try:
        tf.reset_default_graph()
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess, run_name="chat-bot")
        text = gpt2.generate(sess, run_name="chat-bot", temperature=1.0, return_as_list=True, prefix=context,)[0]

        print("---------------------------------------------------\nGenerated is\n" + text)

        lines = len(context.splitlines())
        print("Lines in context = " + str(lines))
        print("Lines in generated = " + str(len(text.splitlines())))
        response = text.splitlines()[lines]

        print("\n---------------------------------------------------\nResponse is:\n"+response)
        print("Adding to conversation\n---------------------------------------------------")
        conversation_file = open('conversation.txt', 'a')
        conversation_file.write(response + "\n")
        conversation_file.close()

    except Exception as e:
        # IF FAIL, REPORT IT
        print(str(e))
        return sendError("An error occured during processing. Contact @BroMan014#7052...\n\n" + str(e)[0:1800] + "\n ...")



    # RETURN RESPONSE
    return response


def sendError(error):
    # CREATE EMBED
    embed = discord.Embed(title="An error occured",
                                                description=error, color=0xff0000)
    # RETURN EMBED
    return embed


client = MyClient()
client.run('ODg0OTUyODAwMDcxNjc1OTU1.YTf-hQ.gCp38OR-3cI4oGveLW8z2ThOK8M')
