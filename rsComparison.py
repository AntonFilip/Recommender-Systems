import gzip
import random
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd

def extractItemRatingsPerUser(dataset):
    itemRatingsPerUser = {}
    for (index, user, item, rating) in dataset.itertuples():
        if (user in itemRatingsPerUser):
            itemRatings = itemRatingsPerUser[user]
            itemRatings[item] = rating
        else:
            itemRatingsPerUser[user] = {item: rating}
    return itemRatingsPerUser


def extractUserRatingsPerItem(dataset):
    userRatingsPerItem = {}
    for (index, user, item, rating) in dataset.itertuples():
        if (item in userRatingsPerItem):
            userRatings = userRatingsPerItem[item]
            userRatings[user] = rating
        else:
            userRatingsPerItem[item] = {user: rating}
    return userRatingsPerItem


def predictRating(sample, ratedBy, similar, n, ratingsPerSample):
    similarityRatingSum = 0
    similaritySum = 0
    for idx, (similarSample, rating) in enumerate(sorted(similar.items(), key=lambda x: x[1], reverse=True)):
        if idx == n:
           break
        similarityRatingSum += ratingsPerSample[similarSample][ratedBy] * similar[similarSample]
        similaritySum += abs(similar[similarSample])
    prediction = similarityRatingSum / similaritySum
    return prediction


def findSimilar(sample, samples, ratedBy, ratingsPerSample, meanRatings):
    similar = {}
    if (len(ratingsPerSample[sample]) == 1):
        return similar
    for candidate in samples:
        if candidate == sample or ratedBy not in ratingsPerSample[candidate]:
            continue
        nominator = 0.0
        denominatorSample = 0.0
        denominatorCandidate = 0.0
        for rater in ratingsPerSample[sample]:
            if rater == ratedBy or rater not in ratingsPerSample[candidate]:
                continue
            samplePart = ratingsPerSample[sample][rater] - meanRatings.at[sample, 'meanRating']
            candidatePart = ratingsPerSample[candidate][rater] - meanRatings.at[candidate, 'meanRating']
            nominator += samplePart * candidatePart
            denominatorSample += samplePart ** 2
            denominatorCandidate += candidatePart ** 2
        denominator = sqrt(denominatorSample * denominatorCandidate)
        if (denominator == 0 or nominator < 0):
            continue
        else:
            similar[candidate] = nominator / denominator

    return similar


def cosineDistance(userContent, itemContents):
    dotProduct = 0
    lengthUserVector = 0
    lengthItemVector = len(itemContents)
    for content in userContent:
        userAffinityToContent = userContent[content]
        if (content in itemContents):
            dotProduct += userAffinityToContent * 1
        lengthUserVector += userAffinityToContent ** 2
    if (lengthUserVector == 0):
        return 0
    distance = dotProduct / (sqrt(lengthUserVector * lengthItemVector))
    return distance

    return distance

def testCF(testSamples, samples, meanRatings, ratingsPerSample, ratedBySamples=None):
    predictionCount = 0
    rmseSum = 0
    maeSum = 0
    for idx, sample in enumerate(testSamples):
        sampleMeanRating = meanRatings.at[sample, 'meanRating']
        # print(f"Mean rating for sample {sample} is {sampleMeanRating}")
        samplePredictedRatings = []
        sampleActualRatings = []
        if (ratedBySamples):
            randomRatedBySamples = [ratedBySamples[idx]]
        else:
            randomRatingsCount = 5 if len(ratingsPerSample[sample]) >= 5 else len(ratingsPerSample[sample])
            randomRatedBySamples = [x[0] for x in random.sample(ratingsPerSample[sample].items(), randomRatingsCount)]
        for ratedBy in randomRatedBySamples:
            similar = findSimilar(sample, samples, ratedBy, ratingsPerSample, meanRatings)
            if (similar):
                predictedRating = predictRating(sample, ratedBy, similar, 50, ratingsPerSample)
                samplePredictedRatings.append(predictedRating)
                actualRating = ratingsPerSample[sample][ratedBy]
                sampleActualRatings.append(actualRating)
                # print(f"Sample: {sample} rated by: {ratedBy} with actual rating: {actualRating} and predicted rating: {predictedRating}")
        if (samplePredictedRatings and sampleActualRatings):
            rmse = sqrt(mean_squared_error(sampleActualRatings, samplePredictedRatings))
            rmseSum += rmse
            mae = mean_absolute_error(sampleActualRatings, samplePredictedRatings)
            maeSum += mae
            # print(f"Root mean squared error is: {rmse} and Mean absolute error is: {mae}")
            predictionCount += 1
    print(f"Result for {predictionCount} predictions. Mean rmse is: {rmseSum/predictionCount} and Mean absolute error is: {maeSum/predictionCount}")

def extractItemContents(contentDictionary):
    itemsContents = {}
    for metadata in contentDictionary:
        itemId = metadata['asin']
        categoriesMatrix =  metadata['categories']
        for categories in categoriesMatrix:
            for category in categories:
                if (itemId not in itemsContents):
                    itemsContents[itemId] = set()
                itemsContents[itemId].add(category)

    return itemsContents


def testUserUserCF(dataset):
    meanRatingsPerUser = pd.DataFrame(dataset.groupby('userId')['rating'].mean())
    meanRatingsPerUser.rename(columns={'rating': 'meanRating'}, inplace=True)
    users = list(dataset.userId.unique())
    itemRatingsPerUser = extractItemRatingsPerUser(dataset)
    testUsers = random.sample(users, 500)
    testCF(testUsers, users, meanRatingsPerUser, itemRatingsPerUser)


def testItemItemCF(dataset):
    meanRatingsPerItem = pd.DataFrame(dataset.groupby('itemId')['rating'].mean())
    meanRatingsPerItem.rename(columns={'rating': 'meanRating'}, inplace=True)
    items = list(dataset.itemId.unique())
    userRatingsPerItem = extractUserRatingsPerItem(dataset)
    testItems = random.sample(items, 500)
    testCF(testItems, items, meanRatingsPerItem, userRatingsPerItem)


def constructUserContent(user, itemRatingsPerUser, itemsContents, userMeanRating):
    userContentRatings = {}
    userContentCount = {}
    userRatedItems = itemRatingsPerUser[user].keys()
    for item in userRatedItems:
        for content in itemsContents[item]:
            if content not in userContentRatings:
                userContentRatings[content] = 0
                userContentCount[content] = 0
            userContentRatings[content] += itemRatingsPerUser[user][item] - userMeanRating
            userContentCount[content] += 1
    userContent = {}
    for content in userContentRatings:
        userContent[content] = userContentRatings[content] / userContentCount[content]
    return userContent


def testContentPrediction(testUsers, meanRatings, itemsContents, itemRatingsPerUser, testItems=None):
    predictionCount = 0
    rmseSum = 0
    maeSum = 0
    for idx, user in enumerate(testUsers):
        userMeanRating = meanRatings.at[user, 'meanRating']
        userContent = constructUserContent(user, itemRatingsPerUser, itemsContents, userMeanRating)
        samplePredictedRatings = []
        sampleActualRatings = []
        if (testItems):
            randomItemsRatedByUser = [testItems[idx]]
        else:
            randomRatingsCount = 5 if len(itemRatingsPerUser[user]) >= 5 else len(itemRatingsPerUser[user])
            randomItemsRatedByUser = [x[0] for x in random.sample(itemRatingsPerUser[user].items(), randomRatingsCount)]
        for item in randomItemsRatedByUser:
            similarity = cosineDistance(userContent, itemsContents[item])
            if (similarity > 0):
                prediction = userMeanRating + similarity * (5 - userMeanRating)
            else:
                prediction = userMeanRating + similarity * (userMeanRating - 1)
            samplePredictedRatings.append(prediction)
            sampleActualRatings.append(itemRatingsPerUser[user][item])

        rmse = sqrt(mean_squared_error(sampleActualRatings, samplePredictedRatings))
        rmseSum += rmse
        mae = mean_absolute_error(sampleActualRatings, samplePredictedRatings)
        maeSum += mae
        # print(f"Root mean squared error is: {rmse} and Mean absolute error is: {mae}")
        predictionCount += 1
    print(f"Result for {predictionCount} predictions. Mean rmse is: {rmseSum/predictionCount} and Mean absolute error is: {maeSum/predictionCount}")

def testContentBased(dataset, itemsContents):
    meanRatingsPerUser = pd.DataFrame(dataset.groupby('userId')['rating'].mean())
    meanRatingsPerUser.rename(columns={'rating': 'meanRating'}, inplace=True)
    itemRatingsPerUser = extractItemRatingsPerUser(dataset)
    users = list(dataset.userId.unique())
    testUsers = random.sample(users, 500)
    testContentPrediction(testUsers, meanRatingsPerUser, itemsContents, itemRatingsPerUser)

def identicalSamplesTest(dataset, itemsContents):
    users = list(dataset.userId.unique())
    testUsers = random.sample(users, 500)
    itemRatingsPerUser = extractItemRatingsPerUser(dataset)
    testItems = [list(usersItemRatings.keys())[0] for usersItemRatings in [itemRatingsPerUser[user] for user in testUsers]]
    meanRatingsPerUser = pd.DataFrame(dataset.groupby('userId')['rating'].mean())
    meanRatingsPerUser.rename(columns={'rating': 'meanRating'}, inplace=True)
    testCF(testUsers, users, meanRatingsPerUser, itemRatingsPerUser, ratedBySamples=testItems)

    items = list(dataset.itemId.unique())
    meanRatingsPerItem = pd.DataFrame(dataset.groupby('itemId')['rating'].mean())
    meanRatingsPerItem.rename(columns={'rating': 'meanRating'}, inplace=True)
    userRatingsPerItem = extractUserRatingsPerItem(dataset)
    testCF(testItems, items, meanRatingsPerItem, userRatingsPerItem, ratedBySamples=testUsers)

    testContentPrediction(testUsers, meanRatingsPerUser, itemsContents, itemRatingsPerUser, testItems=testItems)


def parseContent(path):
    file = open(path, 'r')
    for l in file:
        yield eval(l)

if __name__ == '__main__':
    dataset = pd.read_csv('ratings_Video_Games.csv')
    dataset.drop('timestamp', 1, inplace=True)
    contentDictionary = parseContent('meta_Video_Games.json')
    itemContents = extractItemContents(contentDictionary)
    testUserUserCF(dataset)
    testItemItemCF(dataset)
    testContentBased(dataset, itemContents)
    identicalSamplesTest(dataset, itemContents)
