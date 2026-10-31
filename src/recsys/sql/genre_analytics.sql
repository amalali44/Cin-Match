SELECT genre, COUNT(*) AS rating_count,
       COUNT(DISTINCT userId) AS user_count,
       AVG(rating) AS mean_rating
FROM (
    SELECT r.userId, r.rating, EXPLODE(m.genres) AS genre
    FROM training_ratings r JOIN movie_catalog m USING (movieId)
)
WHERE genre <> '(no genres listed)'
GROUP BY genre
