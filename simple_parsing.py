##############################################
##############################################
##############################################
################## Parsing ###################
##############################################
##############################################
##############################################

from urllib.request import urlopen
from bs4 import BeautifulSoup

# useful function
def lookup_in_dict_vals(v, d):
    for key,val in d.items():
        if v in val:
            return key
    return np.nan
    
def get_text_until_first_s(string, s):
    tmp = ''
    for l in string:
        if l != s:
            tmp +=l
        else:
            break
    return tmp

def get_text_after_nth_s(string, s, n):
    tmp = ''
    n_tmp = 0
    for l in string:
        if l == s:
            n_tmp +=1
        if n_tmp >= n:
            tmp+=l
    return tmp

if __name__ == "__main__":
    

    # this is the website from which i download broadcast information
    url = "https://www.sportsmediawatch.com/nfl-tv-schedule-2021-fox-nbc-cbs-espn-tnf-snf-mnf"  


    page = urlopen(url).read()
    soup = BeautifulSoup(page)


    # get both weeks and days in the order that
    # they are on the website 
    tmp_placeholder = []
    for i in soup.find_all('p'):
        for j in i.find_all('span', {'class' : "sectionhed"}):
            tmp_placeholder.append(j.string)
        for j in i.find_all('span', {'class' : "bold"}):
            tmp_placeholder.append(j.string)
    #     week_date_d[key] = date_list 



    # separate days and weeks
    weeks = [i for i in tmp_placeholder if ('Week' in i)|(i == 'Preseason')]
    days = [i for i in tmp_placeholder if (('Week' not in i)&(i != 'Preseason'))]

    # match days and weeks:
    # first find distance between weeks, which gives us
    # the number of days when broadcasts happen
    tmp_placeholder_bin = [int(('Week' in j)|(j == 'Preseason')) for j in tmp_placeholder]
    dist_list = []
    dist = 0
    for n,i in enumerate(tmp_placeholder_bin):

        if i == 1:
            dist_list.append(dist)
            dist = 0
        else: 
            dist += 1  
            if n == len(tmp_placeholder_bin) - 1:
                dist_list.append(dist)

    dist_list = dist_list[1:]

    # save store as many days for each week
    # as there are to next week
    week_days_dict = {}
    days_copy = days.copy()
    for n,d in enumerate(dist_list):
        key = weeks[n]
        days_for_key = days_copy[:d]
        days_copy = days_copy[d:]
        if key in week_days_dict.keys():
            week_days_dict[key] = week_days_dict[key] + days_for_key
        else:
            week_days_dict[key] = days_for_key


    # parse tables themselves. crucial that days uniquely determine table
    # i fixed in a bruteforce way one case where it wasnt the case
    # iterate over unique days of broadcasting

    n = 0
    df_placeholder = []
    for i in soup.find_all('table', {'style' : "width: 55%;"}): # gets all the tables that have info we want
        t = i.prettify( formatter="html")
        df = pd.read_html(t)[0]
        headers = df.iloc[0]
        new_df  = pd.DataFrame(df.values[1:], columns=headers) # create a header row

        if new_df['Time ET'][0] == 'TBD': # remove one redundant table
            pass

        else:
            new_df['day'] = days[n]
            new_df['week'] = lookup_in_dict_vals(days[n], week_days_dict) # get game week for this day
            df_placeholder.append(new_df)
            n += 1

    df = pd.concat(df_placeholder)
    df = df.reset_index(drop = True)

    # post processing 
    df = df[df['week'] != 'Preseason']
    df['week'] = df['week'].apply(lambda x: x[5:])
    df['week'] = df['week'].astype(int)
    df['TV'] = df['TV'].apply(lambda x: get_text_until_first_s(x, ','))
    df['Game'] = df['Game'].apply(lambda x: get_text_until_first_s(x, '('))
    df['Team_A'] = df['Game'].apply(lambda x: get_text_until_first_s(x, ' '))
    df['Team_B'] = df['Game'].apply(lambda x: get_text_after_nth_s(x,' ', 2))
    df['Team_B'] = df['Team_B'].apply(lambda x: ''.join(e for e in x if e.isalnum()))

    # !!!!!
    # I did some handcleaning in the end by fixing PatriotsBradyreturntoNE and BuccaneersNFLKICKOFFGAME

    df.to_csv('game_data_with_channel.csv', index = False)