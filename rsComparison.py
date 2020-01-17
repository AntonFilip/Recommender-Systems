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


def testCF(testSamples, samples, meanRatings, ratingsPerSample):
    predictionCount = 0
    rmseSum = 0
    maeSum = 0
    for sample in testSamples:
        sampleMeanRating = meanRatings.at[sample, 'meanRating']
        # print(f"Mean rating for sample {sample} is {sampleMeanRating}")
        samplePredictedRatings = []
        sampleActualRatings = []

        randomRatingsCount = 5 if len(ratingsPerSample[sample]) >= 5 else len(ratingsPerSample[sample])
        randomSampleRatings = random.sample(ratingsPerSample[sample].items(), randomRatingsCount)
        for ratedBy, value in randomSampleRatings:
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


def testUserUserCF(dataset):
    meanRatingsPerUser = pd.DataFrame(dataset.groupby('userId')['rating'].mean())
    meanRatingsPerUser.rename(columns={'rating': 'meanRating'}, inplace=True)
    users = list(dataset.userId.unique())
    itemRatingsPerUser = extractItemRatingsPerUser(dataset)
    testUsers = random.sample(users, 100)
    testCF(testUsers, users, meanRatingsPerUser, itemRatingsPerUser)


def testItemItemCF(dataset):
    meanRatingsPerItem = pd.DataFrame(dataset.groupby('itemId')['rating'].mean())
    meanRatingsPerItem.rename(columns={'rating': 'meanRating'}, inplace=True)
    items = list(dataset.itemId.unique())
    userRatingsPerItem = extractUserRatingsPerItem(dataset)
    testItems = random.sample(items, 100)
    testCF(testItems, items, meanRatingsPerItem, userRatingsPerItem)


if __name__ == '__main__':
    dataset = pd.read_csv('ratings_Video_Games.csv')
    dataset.drop('timestamp', 1, inplace=True)
    testUserUserCF(dataset)
    testItemItemCF(dataset)
