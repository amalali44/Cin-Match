SELECT userId,
       COUNT(*) AS user_rating_count,
       AVG(rating) AS user_mean_rating,
       MAX(timestamp) AS user_last_timestamp
FROM training_ratings
GROUP BY userId
