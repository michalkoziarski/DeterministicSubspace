from trial import run_trial
from database import pull_pending


while True:
    trial = pull_pending()

    if trial is None:
        break

    run_trial(trial)
