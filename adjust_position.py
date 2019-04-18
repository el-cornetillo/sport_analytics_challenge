def adjust_prediction(last_event, lbo_event, last_x, last_y, pred_x, pred_y):
    if last_event == "aerial":
        if lbo_event != last_event:
            return (last_x, last_y)
    if last_event == "dispossessed":
        return (last_x, last_y)
    if ((last_event == "challenge") and (lbo_event != "take_on")):
        return (last_x, last_y)
    if ((last_event == "take_on") and (lbo_event == "pass")):
        return (last_x, last_y)
    if last_event == "corner_awarded":
        if lbo_event != last_event:
            return (last_x, last_y)
        else:
            corner_pos_x = .5 if last_x < 50 else .95
            corner_pos_y = .5 if last_y < 50 else .95
            return (corner_pos_x, corner_pos_y)
    if last_event == "foul":
        if lbo_event != last_event:
            return (last_x, last_y)
    if last_event == "foul_throw-in":
        if lbo_event != last_event:
            return (100 - last_x, 100 - last_y)
    if last_event == "goal":
        return (50, 50)
    if last_event == "penalty_faced":
        if lbo_event in ['card', 'contentious_referee_decision', 'foul']:
            if ((last_x == 100) and (last_y == 100)):
                return (88.5, 50)
            else:
                return (11.5, 50)
        if lbo_event == "goal":
            return (50, 50)
    if last_event == "player_off":
        return (last_x, last_y)
    if last_event == "referee_drop_ball":
        if lbo_event != last_event:
            return (100 - last_x, 100 - last_y)

    if last_event == "out":
        if lbo_event != last_event:
            if ((last_x < 0) or (last_x > 100)):
                return (last_x, pred_y)
            else:
                return (last_x, last_y)
        if lbo_event == last_event:
            if not ((last_x < 0) or (last_x > 100)):
                return (pred_x, 0) if last_y < 0 else (pred_x, 100)

    return max(min(pred_x, 102), -2), max(min(pred_y, 102), -2)