import discord
import gpt_2_simple as gpt2
import os
import requests
import sys
import tensorflow as tf
from random import randint
import re
import json
from googletrans import Translator
import yelp_photo_generation

class MyClient(discord.Client):

    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))
        channel = client.get_channel(642818419007422475)
        await client.change_presence(activity=discord.Activity(
                                        type=discord.ActivityType.watching,
                                        name="the chat. Type !help for help"))
        # await channel.send('I\'m online')

    async def on_message(self, message):
        recieved = message.content
        channel = message.channel
        if not message.author.bot:
            await parseCommands(client, channel, message)


async def parseCommands(client, channel, message):
    # SET STATUS TO GENERATING
    await client.change_presence(activity=discord.Game(
                                            name="a generated response..."))
    # Get type of command
    command = message.content.partition(' ')[0].lower()
    patternFull = re.compile("!generate-(\d{1,2})")
    patternShort = re.compile("!g-(\d{1,2})")
    patternYelp = re.compile("!yelp-(\d{1,2})")
    if command == "!generate" or command == "!g":
        # RESPOND WITH THE GENERATED TEXT
        async with channel.typing():
            await channel.send(embed=sendGeneratedResponse(message))

    elif patternYelp.match(command):
        # GET THE NUMBER TO GENERATE
        match = re.search('!yelp-(\d{1,2})', command, re.IGNORECASE)
        num = int(match.group(1))
        for x in range(num):
            async with channel.typing():
                await channel.send(file=sendYelpPhoto(message))

    elif patternFull.match(command):
        # GET THE NUMBER TO GENERATE
        match = re.search('!generate-(\d{1,3})', command, re.IGNORECASE)
        num = int(match.group(1))
        for x in range(num):
            # RESPOND WITH THE GENERATED TEXT
            async with channel.typing():
                await channel.send(embed=sendGeneratedResponse(message))

    elif patternShort.match(command):
        # GET THE NUMBER TO GENERATE
        match = re.search('!g-(\d{1,3})', command, re.IGNORECASE)
        num = int(match.group(1))
        for x in range(num):
            # RESPOND WITH THE GENERATED TEXT
            async with channel.typing():
                await channel.send(embed=sendGeneratedResponse(message))

    elif command == "!sets":
        # RESPOND WITH THE SETS
        async with channel.typing():
            messages = sendSets()
            for message in messages:
            	await channel.send(embed=message)

    elif command == "!commands":
        # RESPOND WITH THE SETS
        async with channel.typing():
            await channel.send(embed=discord.Embed(title="Available Commands:",
                                description=commandList(), color=0x00ff55))

    elif command == "!help":
            # RESPOND WITH THE SETS
        async with channel.typing():
            await channel.send(embed=sendHelp())

    elif command.startswith("!"):
            # RESPOND WITH THE SETS
        async with channel.typing():
            await channel.send(embed=discord.Embed(
                title=" That command wasn't recognized. Available Commands:",
                description=commandList(), color=0xff9000))
    # SET STATUS TO WAITING
    await client.change_presence(activity=discord.Activity(
                                        type=discord.ActivityType.watching,
                                        name="the chat. Type !help for help"))

def sendYelpPhoto(message):
    print("Sending photo!")
    #get the prefix
    prefix = re.search('\"(.+)\"', message.content).group(1)
    # generate photo
    yelp_photo_generation.generate(prefix)
    # send the newly generated tmp.png
    return discord.File(fp='tmp.png', filename='yelp-review.png')

def sendSets():
    # BUILD THE SETS MESSAGE FROM CONFIG
    # LOAD THE CONFIG FILE
    with open('config.json') as f:
        configs = json.load(f)
    # CREATE THE EMBEDs
    messages = []
    # ITERATE THOUGH THE LIST OF SETS
    for set in configs['sets']:
        # GET THE NAME, DESCRIPTION, AND RESULT FOR EACH SET
        name = "**" + configs[set]['name'] + "**"
        value = ">>> " + configs[set]['description'] +  "\n *" + configs[set]['result'] + "*"
        # ADD TO THE EMBED
        message = discord.Embed()
        message.add_field(name=name, value=value, inline=True)
        messages.append(message)

    # RETURN THE LIST
    return messages


def commandList():
    list = "`!g(enerate) ({set}) \"{prefix}\"`\nPrompts the Bot to generate some text from the given sample set `set` and *optionally* with a provided `prefix`\n\n"
    list = list + \
            "`!g(enerate)-{N} ({set}) \"{prefix}\"`\nWorks the same way as the vanilla `generate` command, but will create `N` number of samples with the provided parameters\n\n"
    list = list + "`!commands`\nDisplays a list of commands\n\n"
    list = list + "`!sets`\nDisplays a list of trained sample sets you can reference in `generate`\n\n"
    list = list + "`!help`\nDisplays this message\n\n"
    return list


def sendHelp():
    # BUILD THE SETS MESSAGE FROM CONFIG
    # CREATE THE EMBED
    message = discord.Embed(
            title="GPT-2 Discord bot by @BroMan014#7052:", color=0x00ff55)
    # CREATE THE CONTENT
    generalInfo = "GPT bot is a python discord bot that generates samples using the GPT-2 language model.\n"
    generalInfo = generalInfo + \
            "GPT bot uses discord.py to handle discord communications: https://discordpy.readthedocs.io/en/latest/\n"
    generalInfo = generalInfo + \
            "GPT bot uses gpt_2_simple to train and generate sample sets: https://github.com/minimaxir/gpt-2-simple\n"
    generalInfo = generalInfo + \
            "For more info on how GPT-2 works: https://openai.com/blog/better-language-models/\n"
    generalInfo = generalInfo + \
            "Source for this bot lives here: https://bitbucket.org/BrodyHarris/gpt-2-uses/src/master/\n"
    generalInfo = generalInfo + "For suggestions on sample sets, DM @BroMan014#7052\n"
    tips = "- Providing prefixes that make more sense in the context of the sample set will achieve better results.\n"
    tips = tips + "- Generating 2 or 3 results at a time can be very helpful if looking for a desired result.\n"
    tips = tips + \
            "- You can forego the prefix functionality by sending only `generate (trump-tweet)`, and the bot will drum up a result all on it's own.\n"
    tips = tips + "- The bot processes in a LIFO fashion and will not work synchronously.\n"
    # ADD TO THE EMBED
    message.add_field(name="General Info", value=generalInfo, inline=False)
    message.add_field(name="Commands", value=commandList(), inline=True)
    message.add_field(name="Tips", value=tips, inline=True)
    # RETURN THE LIST
    return message


def sendGeneratedResponse(message):
    # GET THE SET REQUESTED FROM
    set = re.search('[(](.+)[)]', message.content).group(1).lower()

    # GET THE PREFIX TEXT, IF ANY
    try:
        prefix = re.search('\"(.+)\"', message.content).group(1)
    except:
        prefix = ""
    # TRY TO LOAD CONFIGS
    try:
        with open('config.json') as f:
            configs = json.load(f)
            test = configs[set]
        # IF NOT FOUND, MODEL MUST NOT EXIST
    except:
        return sendError("Couldn't find that set in the data.\n **Are you sure you typed it correctly? You can run `list-sets` to see what sets are available.**")

    # CREATE TITLE PLACEHOLDER
    title = ""
    # GET TITLE DIMENTIONS FOR DYNAMIC TITLES
    titleDimentions = configs[set]['title-dimentions']
    # IF DYNAMIC
    if titleDimentions > 0:
        # GET EACH PART
        for part in range(titleDimentions):
            # GRAB INDEX
            part = part + 1
            # GET NUMBER OF COMPONENTS TO THIS PART
            components = len(configs[set]["title-part-" + str(part)])
            # GET A RANDOM FROM THE INDEX
            randpiece = randint(0, components-1)
            # PULL A RANDOM COMPONENT FROM THIS PIECE AND ADD IT TO THE TITLE
            title = title + configs[set]["title-part-" + str(part)][randpiece]
    else:
        # IF NOT DYNAMIC JUST GET THE DEFAULT
        title = configs[set]['embed-title']
    # GRAB THE THUMBNAIL
    thumb = configs[set]['embed-thumb-url']
    # GRAB THE EMBED-COLOR
    color = discord.Colour(int(configs[set]['embed-color'], 16))
    # GRAB THE CUSTOM PREFIX, IF ANY
    setPrefix = configs[set]['prefix']

    # TRY THE PROCESSING
    try:
        tf.reset_default_graph()
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess, run_name=set)
        #await channel.send("Generating "+ model +" that starts with \n\"" + prefix +"\"\n")
        text = gpt2.generate(sess, run_name=set, temperature=1.0, return_as_list=True,
                                                 prefix=setPrefix + " " + prefix, truncate='<|endoftext|>', include_prefix=True,)[0]
    except Exception as e:
        # IF FAIL, REPORT IT
        print(str(e))
        return sendError("An error occured during processing. Contact @BroMan014#7052...\n\n" + str(e)[0:1800] + "\n ...")

    # Translate result
    #translator = Translator()
    #translation = translator.translate(text, dest='en')
    #text = translation.text

    # FORMAT RESULT
    initialText = text[0:1800]
    initialText = initialText.replace('<|startoftext|>', '')
    initialText = re.sub(r"\(\d{0,2}:\d{2}\)", "", initialText)
    initialText = re.sub(r"\\n", "\n", initialText)
    initialText = initialText.replace('""','"')

    # CREATE EMBED TO SEND
    if re.match("wisdom", set):
        embed = discord.Embed(title=prefix, description=re.sub(r'Question.*','', re.sub(r'.*Answer:','',initialText)), color=color)
    else:
        embed = discord.Embed(title=title, description=initialText, color=color)
    embed.set_thumbnail(url=thumb)
    # ADD TO THE EMBED
    embed.set_footer(text="Generated using the " + set
                                     + " sample set. Run !help for more info.")
    # RETURN EMBED
    return embed


def sendError(error):
    # CREATE EMBED
    embed = discord.Embed(title="An error occured",
                                                description=error, color=0xff0000)
    # RETURN EMBED
    return embed


client = MyClient()
client.run('NjQyMTY4MTE0NTc0MDAwMTQx.XcS_qQ.ALHOhk-6QNQuZ0K1Q5g114c4a2A')
