#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:54:38 2024

@author: safayat
"""


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
paragraph ="""
Once upon a time, in a faraway kingdom nestled amidst rolling hills and enchanted forests, there lived a humble woodcutter named Cedric. He dwelled in a cozy cottage at the edge of the Great Whispering Woods, where ancient trees whispered secrets of the past to those who dared to listen.

Cedric was known throughout the kingdom for his gentle heart and unwavering kindness. Every morning, he would set out into the woods with his trusty axe, harvesting firewood to sell at the market. Despite his meager earnings, Cedric always shared what little he had with those in need, offering food to hungry travelers and shelter to lost souls.

One crisp autumn morning, as Cedric ventured deeper into the woods than usual, he stumbled upon a hidden glade bathed in golden sunlight. At its center stood a majestic oak tree, its branches adorned with shimmering leaves of every hue. Beneath the tree sat a tiny creature, no larger than a squirrel, with delicate wings that sparkled like dewdrops in the morning light.

The creature introduced herself as Willow, a woodland sprite tasked with guarding the ancient oak tree. She explained that the tree was enchanted, its roots intertwined with the very heartbeat of the forest. But now, a dark shadow loomed over the land, threatening to extinguish the tree's magic forever.

With a heavy heart, Cedric pledged to help Willow protect the sacred oak. Together, they embarked on a quest to unravel the mystery of the encroaching darkness and restore balance to the forest.

Their journey led them through dark caverns and treacherous swamps, where they encountered mythical creatures both friend and foe. Along the way, they formed unlikely alliances with a mischievous pixie, a wise old owl, and a band of merry dwarves, each offering their own unique talents to aid in their quest.

"""
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)   
    
    
    