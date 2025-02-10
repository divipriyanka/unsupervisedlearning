# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:57:22 2025

@author: priya
"""

import numpy as np
import pickle
import streamlit as st
import difflib
import pandas as pd

# Load your precomputed movie data and similarity matrix
movies_data = pickle.load(open('movies_data.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def movie_recommender_system(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return ["No close match found. Please try another movie."]

    close_match = find_close_match[0]
    index_of_the_movie = index_of_the_movie = movies_data[movies_data.title == close_match]['mID'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, movie in enumerate(sorted_similar_movies[1:11], start=1):  # Skip first because it's the input movie itself
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
