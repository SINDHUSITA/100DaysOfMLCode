## Predicting the number of Covid-19 cases

The amount of daily increase in the number of Covid-19 cases is observed to be in an exponential trend.

Link to dataset: [Here](https://raw.githubusercontent.com/WidhyaOrg/datasets/master/covid19.csv)

To predict the number of confirmed cases on a particular day, we use an exponential function.

Number of cases(forecasted) = (Number of observed cases) * (e^(rate of increase * time period))

Rate of increase over a time period is nothing but average of rate incerease per day.

This is calculated as, Rate of increase = (Number of cases on next day - Number of cases on current day) / Number of cases on current day
