import generate
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import requests
import json
import random
from datetime import date
from faker import Faker
fake = Faker()
import re
import urllib.request

def generate(prefix):
    text = generate.generate("yelp-review", prefix)
    withlines = text.replace("\\n", "\n")

    msg = withlines
    msg = msg.replace("\\/", "/")
    msg = msg.replace("\\", "...")

    def text_wrap(text, font, max_width):
            """Wrap text base on specified width.
            This is to enable text of width more than the image width to be display
            nicely.
            @params:
                text: str
                    text to wrap
                font: obj
                    font of the text
                max_width: int
                    width to split the text with
            @return
                lines: list[str]
                    list of sub-strings
            """
            lines = []
            text = text.replace("\n", " \n ")
            pattern = re.compile("\n")
            # If the text width is smaller than the image width, then no need to split
            # just add it to the line list and return
            if font.getsize(text)[0]  <= max_width:
                lines.append(text)
            else:
                #split the line by spaces to get words
                words = text.split(' ')
                i = 0
                # append every word to a line while its width is shorter than the image width
                while i < len(words):
                    line = ''
                    while i < len(words) and font.getsize(line + words[i])[0] <= max_width and not pattern.match(words[i]):
                        line = line + words[i]+ " "
                        i += 1
                    if not line:
                        line = words[i]
                        i += 1
                    lines.append(line)
                    if line == "\n":
                         lines.remove(line)
                    elif not re.match(r'\w', line):
                         lines[lines.index(line)] = line[1:]
            return lines

    review = msg

    ######## GENERATE THE IMAGE
    fnt = ImageFont.truetype('font-yelp.ttf', 20)
    datefnt = ImageFont.truetype('font-yelp.ttf', 15)
    locfnt = ImageFont.truetype('font-yelp.ttf', 10)
    lines = text_wrap(review, fnt, 500)
    width = 800
    filename = "tmp.png"
    lineheight = fnt.getsize(lines[0])[1] + 3
    height = len(lines) * (lineheight) + 75
    if height < 150:
        height = 150
    image = Image.new(mode = "RGB", size = (width,height), color = "white")
    draw = ImageDraw.Draw(image)
    for line in lines:
        draw.text((275,50 + lineheight * lines.index(line)), line, font = fnt, fill = (0,0,0))

    # Rating Image
    found_rating = False
    review_lower = review.lower()
    m = re.findall('(\d|no|all|one|two|three|four|five) star', review_lower)
    if m:
        rating = m[0]
        if str(rating) == "no":
            rating = 1
        elif str(rating) == "all":
            rating = 5
        elif str(rating) == "one":
            rating = 1
        elif str(rating) == "two":
            rating = 2
        elif str(rating) == "three":
            rating = 3
        elif str(rating) == "four":
            rating = 4
        elif str(rating) == "five":
            rating = 5
        elif int(rating) < 1:
            rating = 1
        elif int(rating) > 5:
            rating = 5
        found_rating = True
    else:
        url = "https://japerk-text-processing.p.rapidapi.com/sentiment/"
        encodedreview = review.encode('utf-8')
        payload = "language=english&text=" + str(encodedreview)
        headers = {
            'x-rapidapi-host': "japerk-text-processing.p.rapidapi.com",
            'x-rapidapi-key': "928ca024c2mshbdbe2d79ff1b355p189954jsnff7f62b17eb7",
            'content-type': "application/x-www-form-urlencoded"
            }

        response = requests.request("POST", url, data=payload, headers=headers)

        print(response.text)

        data = json.loads(response.text)
        pos_score = (data["probability"].get("pos"))  + .05
        pos_perc = int(pos_score * 100)
        if pos_perc > 100:
            pos_perc = 100
        positivity = str(pos_perc) + "% positive"
        rating = 3;
        if 0 <= pos_perc < 30:
            rating = 1
        elif 30 <= pos_perc < 40:
            rating = 2
        elif 40 <= pos_perc < 50:
            rating = 3
        elif 50 <= pos_perc < 70:
            rating = 4
        elif 70 <= pos_perc <= 100:
            rating = 5
        print(positivity)
        print(rating)
    rating_img = Image.open("Cropped" + str(rating) + ".PNG")
    image.paste(rating_img, (275, 20))

    #date
    today = str(date.today())
    draw.text((405, 23), today, font = datefnt, fill = (100,100,100))

    ## name
    name = fake.name_male()
    draw.text((140, 23), name, font = datefnt, fill = (24, 132, 153))

    ##location
    location = re.sub("\d+", " ", fake.address().partition('\n')[2])
    draw.text((140, 45), location, font = locfnt, fill = (100,100,100))

    ## Photoif (
    facenum = random.randint(0,199)
    facegender = "men"
    if facenum > 99:
       facegender = "women"
       facenum = facenum - 100
    faceurl = "https://randomuser.me/api/portraits/med/" + facegender + "/" + str(facenum) + ".jpg"
    print(faceurl)
    urllib.request.urlretrieve(faceurl, "tmpface")
    face = Image.open("tmpface").resize((100,100))
    image.paste(face, (25, 20))

    image.save(filename)
    return image
