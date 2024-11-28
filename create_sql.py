import datetime

file_name = "holidays.txt"
sql_script = "insert_dates.sql"

obj = {}

with open(file_name, 'r') as f:
    # iterate each line in the file and create dictionary of all the holidays in the format
    # {'2020-01-01' : 'New Year's Day', ...}
    for line in f:
        arr = line.strip().split(",")
        date = arr[0]
        description = arr[1]
        obj[date] = description
    
    # iterate the KEYS and get the VALUE
    for x in obj:
        date_arr = str(x).split("-")
        year = int(date_arr[0])
        month = int(date_arr[1])
        day = int(date_arr[2])
        
        doy = datetime.datetime(year,month,day).timetuple().tm_yday
        # print(f'{x},{obj[x]},{str(doy)}')

    f.close()
    
table_name = "Calendar"

with open(sql_script, "a") as f:
    for year in range(1992, 2021):
        for day in range(1, 367 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 366):  # Handle leap year
            current_date = datetime.datetime.strptime(str(year) + "-" + str(day), "%Y-%j").strftime("%Y-%m-%d")
            if current_date in obj:
                description = obj[current_date]
                f.write(f"INSERT INTO {table_name} (DOY, Year, Date, Description) VALUES ({day}, {year}, {current_date}, \"{description}\")\n")
            else:
                f.write(f"INSERT INTO {table_name} (DOY, Year, Date, Description) VALUES ({day}, {year}, {current_date}, NULL)\n")
                
    f.close()

    
        