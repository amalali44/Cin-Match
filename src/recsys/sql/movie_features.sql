SELECT movieId,
       COUNT(*) AS movie_rating_count,
       AVG(rating) AS movie_mean_rating
FROM training_ratings
GROUP BY movieId
