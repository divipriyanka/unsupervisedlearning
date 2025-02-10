# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:16:59 2025

@author: priya
"""

import streamlit as st
import numpy as np
import pickle
import difflib
import pandas as pd    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity    

def movie_recommender_system(movie_name):
    st.title("Movie Based Recommendation System")

    # Load data
    movies_data = pd.read_csv("movies.csv")
    
    #selecting the relevant features for recommendation
    selected_features = ['genres','keywords','tagline','cast','director']
    print(selected_features)
    
    #replacing the null values with null string

    for feature in selected_features :
        movies_data[feature] = movies_data[feature].fillna('')    
        
    # Combining all the 5 selected features
    combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
    
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return ["No close match found. Please try another movie."]

    
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
    print('Movies suggested for you : \n')


    recommendations = []
    for i, movie in enumerate(sorted_similar_movies[1:31], start=1):  
        index = movie[0]
        recommended_movie = movies_data[movies_data.index == index]['title'].values[0]
        recommendations.append(f"{i}. {recommended_movie}")

    return recommendations

def main():
    # Streamlit App UI
    st.title('Movie Recommender System Web App')

    # User Input
    movie_name = st.text_input('Enter your favorite movie name:')

    if st.button('Movies recommended'):
        if movie_name:
            recommended_movies = movie_recommender_system(movie_name)
            st.write("### Recommended Movies:")
            for movie in recommended_movies:
                st.write(movie)
        else:
            st.write("Please enter a movie name.")

if __name__ == '__main__':
    main()