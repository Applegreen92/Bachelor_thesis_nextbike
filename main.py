import holidays

my_holidays = holidays.Germany(state='NW', years=2023)

for d in sorted(my_holidays):
    print(d)