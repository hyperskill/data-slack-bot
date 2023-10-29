--
WITH

users AS (

    SELECT user_id
    FROM hyperskill_private.users

    WHERE is_guest = 0
        AND id NOT IN (SELECT user_id FROM hyperskill.users_is_biased)
        AND is_testee = 1
        -- c) Not from Russia, Belarus or Ukraine
        -- country_by_ip_id
        AND dictGetString('hyperskill.alt_cities_country_dict', 'name', toUInt64(country_by_ip_id)
            ) NOT IN ('Russia', 'Ukraine', 'Belarus')
        -- JetSales
        AND id NOT IN (
            SELECT user_id
            FROM hyperskill_private.stg_subscriptions__hist
            WHERE JSONExtractString(subscription_data, 'ownerCountry') IN ('RU', 'UA', 'BY')
        )

),

talg AS (

    SELECT
        user_id,
        question_1,
        splitByChar(
                ';',
                dictGetString(
                        'hyperskill.tell_about_your_level_and_goals_constants_dict',
                        'question_1_values', toUInt64(1)
                    )
            )[toInt32(question_1)] AS experience,
        question_2,
        splitByChar(
                ';',
                dictGetString(
                        'hyperskill.tell_about_your_level_and_goals_constants_dict',
                        'question_2_values', toUInt64(1)
                    )
            )[toInt8(question_2)] AS motivation
    FROM hyperskill.polls
    WHERE poll_id = 4
      AND date >= toDate('2022-06-01')

),

result AS (

    SELECT
        user_id,
        dictGetString('hyperskill_private.users_dict', 'email', toUInt64(user_id)) AS email,
        -- current track, if ever solved a stage
        argMax(selected_track_id, dt) AS latest_selected_track_id,
        if(
            countIf(action = 'completed_stage') > 0,
            dictGetStringOrDefault(
                    'hyperskill.tracks_track_dict',
                    'title',
                    toUInt64(latest_selected_track_id),
                    'None'
                ),
            'None'
        ) AS current_track_if_ever_solved_a_stage,
        any(experience) AS talg_experience,
        any(motivation) AS talg_motivation
    FROM hyperskill.content
    INNER JOIN talg
      ON talg.user_id = content.user_id

    WHERE action IN ('completed_stage', 'completed_topic', 'completed_step')
      AND user_id IN (SELECT user_id FROM users) -- a), b), c)
      AND user_id IN (SELECT user_id FROM talg)  -- get only those who answered to the poll
    GROUP BY user_id
    HAVING
        -- d) Solved at least 1 problem in the past two weeks;
        countIf(date, action = 'completed_step' AND step_type <> 'text'
                             AND date BETWEEN today() - 14 AND today() - 1) > 0
        -- e) Solved at least 1 topic overall;
        AND countIf(date, action = 'completed_topic' AND solving_context = 'by_steps') > 0

)

SELECT
    user_id,
    email,
    current_track_if_ever_solved_a_stage,
    talg_experience,
    talg_motivation
FROM result;
