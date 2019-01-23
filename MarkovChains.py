def generateTable(data, k=3):
    T = {}
    for i in range(len(data) - k):
        x = data[i:i + k]
        y = data[i + k]
        if T.get(x) is None:
            T[x] = {}
            T[x][y] = 1
        else:
            if T[x].get(y) is None:
                T[x][y] = 1
            else:
                T[x][y] += 1
    return T


def convertToProbs(T):
    for kx in T.keys():
        s = float(sum(T[kx].values()))
        for k in T[kx].keys():
            T[kx][k] = T[kx][k] / s
    return T


def trainMC(text, k=4):
    T = generateTable(text, k)
    T = convertToProbs(T)
    return T


text = """The Prime Minster, Shri Narendra Modi today addressed the nation from the ramparts of the Red Fort on the 72nd Independence Day. Following are the highlights from his speech
Today, the country is full of self-confidence. The country is scaling new heights by working extremely hard with a resolve to scale new heights.
We have been celebrating this festival of independence, at a time when our daughters from states of Uttarakhand, Himachal, Manipur, Telangana and Andhra Pradesh have come back after circumnavigating the seven seas. They have come back amongst us by (unfurling tricolour in seven seas) turning the seven seas into the colour of our Tricolor.
Our young tribal children, those who live in our forests in the far-flung areas, they have enhanced the glory of the tri-colour by unfurling it on the Mount Everest.
Whether it is a Dalit or be it someone who has been persecuted or exploited or be it a deprived person or women, our Parliament has made the social justice even morestronger with all the sensitivity and alertness to protect their interests.
The demand to give constitutional status to the OBC commission had been raised for years. This time our Parliament has made an effort to protect the interests of backward classes, interests of the extremely backward classes by according the constitutional status to the OBC commission.
I want to reassure the people who have lost their loved ones and are facing a lot of distress due to floods that the country is with them and is making complete efforts for helping them out. My heartfelt condolences are with those who have lost their loved ones.
The next year will mark 100 years of the Jalliwanwallahbagh massacre. The masses had sacrificed their lives for the country's freedom; and the exploitation had crossed all limits. The Jalliwanwallahbagh incident inspires us of the sacrifices made by those brave hearts. I salute all those brave hearts from the bottom of my heart."""

model = trainMC(text)


def sample_next(ctx, T, k=4):
    import numpy as np
    ctx = ctx[-k:]
    if T.get(ctx) is None:
        return ' '
    possible_char = list(T[ctx].keys())
    possible_values = list(T[ctx].values())
    return np.random.choice(possible_char, p=possible_values)


def generateText(start, k=4, maxLen=1000):
    sentence = start
    ctx = start[-k:]
    for ix in range(maxLen):
        next_pred = sample_next(ctx, model, k)
        sentence += next_pred
        ctx = sentence[-k:]
    return sentence


text = generateText('demo', maxLen=50)
print(text)
