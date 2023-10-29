WITH

calculations AS (

    SELECT
        uniqExact(user_id) AS users_total,
        uniqExactIf(user_id, action = 'completed_submission') AS target_users,
        round(target_users / users_total, 2) AS target_users_part
    FROM hyperskill.content
    WHERE date > today() - 28 * 3  -- get period to reflect actual product state
      AND date = registration_date  -- analyze events only during registration day
      AND action IN ('registered_user', 'completed_submission')

),

result AS (

    SELECT
        target_users_part,
        target_users,
        users_total
    FROM calculations

)

SELECT *
FROM result;
