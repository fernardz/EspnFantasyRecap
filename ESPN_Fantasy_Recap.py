import pandas as pd
import requests
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
import random
import json
from io import BytesIO
from datetime import datetime as date
from sklearn.preprocessing import StandardScaler,MaxAbsScaler, MinMaxScaler
import numpy as np
import jwt

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta, date
from airflow.utils.dates import days_ago
from airflow.models import Variable

# Function to get the league data
def get_scoreboard(w, lid):
    query='http://fantasy.espn.com/apis/v3/'+ \
                            'games/ffl/seasons/2020/segments/0/'+ \
                            'leagues/{}?view=mBoxscore&scoringPeriodId={}&matchupPeriodId={}'.format(lid,w,w)
    scoreboard=requests.get(query)
    if scoreboard.status_code==200:
        return scoreboard.json()
    else:
        return 'error'

# We create a class with all our mappings that we need to use
class ffantasy_maps:
    def __init__(self,sc_data):
        self.pos_dict={0:'QB',2:'RB',4:'WR',6:'TE',23:'FLEX',16:'D/ST',17:'K',20:'BNCK'}

        self.Rs={0:1,2:2,4:2,6:1,23:1,16:1,17:1}

        self.team_map={itm['id']:itm['location']+' '+itm['nickname'] for itm in sc_data['teams']}
        self.iteam_map = {v: k for k, v in self.team_map.items()}

        self.divs={0:'East',1:'West',2:'Mid'}

        self.team_div={self.team_map.get(x['id']):
                       self.divs.get(x['divisionId']) for x in sc_data['teams']}

        self.record_dict={self.team_map.get(itm['id']):
                          itm['record'] for itm in sc_data['teams']}

        self.rtxt_dict={k:'({},{},{})'.format(v['overall']['wins'],
                               v['overall']['ties'],
                               v['overall']['losses']) for k,v in self.record_dict.items()}

# Create a DF with all the week data
def create_week_data(sc_data,ffmap,w):
    gnum=1
    roster_array=[]
    summary={}
    for s in sc_data['schedule']:
        if s['matchupPeriodId']==w:
            for ttype in ['home','away']:
                team=s[ttype]
                roster=team['rosterForCurrentScoringPeriod']

                for e in roster['entries']:
                    pscore=e['playerPoolEntry']['appliedStatTotal']
                    pfname=e['playerPoolEntry']['player']['fullName']
                    pslots=e['playerPoolEntry']['player']['eligibleSlots']
                    pslot=e['lineupSlotId']
                    roster_array.append([gnum,ttype,s[ttype]['teamId'],pscore,pfname,pslot,pslots])
            summary[gnum]={'home':s['home']['teamId'],
                           'home_score':s['home']['rosterForCurrentScoringPeriod']['appliedStatTotal'],
                           'home_name':ffmap.team_map.get(s['home']['teamId']),
                           'away':s['away']['teamId'],
                           'away_score':s['away']['rosterForCurrentScoringPeriod']['appliedStatTotal'],
                           'away_name':ffmap.team_map.get(s['away']['teamId']),}

            gnum=gnum+1

    infopd=pd.DataFrame(roster_array,columns=['G','Ttype','TeamId','Score','Name','Slot','Slots'])
    infopd['SlotPos']=infopd.Slot.map(ffmap.pos_dict)
    infopd['TeamName']=infopd.TeamId.map(ffmap.team_map)
    infopd.SlotPos.fillna('Unknown',inplace=True)
    return infopd, summary

# Create all possible scores, for each team
def get_all_pos_scores(infopd,ffmap):
    pos_scores={}
    for g,lpd in infopd.groupby('TeamName'):
        posln={}
        for k in ffmap.Rs:
            posln[k]=lpd[lpd.Slots.apply(lambda x:k in x)].index.values

        tarr=[]
        for i,k in ffmap.Rs.items():
            for ii in range(0,k):
                tarr.append(list(posln[i]))

        lpscore=[]
        #for lpos in itertools.product(posln[0],posln[2],posln[2],posln[4],posln[4],posln[6],posln[23],posln[16],posln[17]):
        for lpos in itertools.product(*tarr):
            if len(set(lpos))==9:
                lpscore.append(lpd.loc[list(lpos)].Score.sum())
        pos_scores[g]=lpscore
    return pos_scores

# Create the headings for each game
def get_headings(summary,ffmap):
    headings={}
    for i, g in summary.items():
        home=g['home_name']
        away=g['away_name']
        home_score=g['home_score']
        away_score=g['away_score']

        home_div=ffmap.team_div.get(home)
        away_div=ffmap.team_div.get(away)

        home_rec=ffmap.rtxt_dict.get(home)
        away_rec=ffmap.rtxt_dict.get(away)

        heading="{} {} {} Div.:{:.1f} vs {} {} {} Div.:{:.1f} \n".format(home,
                                                              home_rec,
                                                              home_div,
                                                              home_score,
                                                              away,
                                                              away_rec,
                                                              away_div,
                                                              away_score)
        headings[i]=heading
    return headings

#Create a simple db with all possible scores
def gen_tds(pos_scores):
    tds=pd.DataFrame([])
    for i,tr in pos_scores.items():
        tdf=pd.DataFrame(tr,columns=['sc'])
        tdf['TeamName']=i
        tds=pd.concat([tds,tdf])
    return tds

#Create all the plots
def gen_game_plots(summary, pos_scores, w):
    gplots={}
    for i,gv in summary.items():
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(15,5))
        sns.histplot(pos_scores[gv['home_name']],ax=ax1,bins=15,stat="probability")
        ax1.axvline(gv['home_score'],color='r')
        ax1.set_title('{}: {:.1f}'.format(gv['home_name'],gv['home_score']))
        sns.histplot(pos_scores[gv['away_name']],ax=ax2,bins=15,stat="probability")
        ax2.axvline(gv['away_score'],color='r')
        ax2.set_title('{}: {:.1f}'.format(gv['away_name'],gv['away_score']))
        fname='data/w{}_g{}_prob.png'.format(w,i)
        plt.savefig(fname)
        gplots[i]={'impath':fname}
    return gplots

#Create overall plot
def gen_overall_chart(tds,infopd, w):
    fb,axbp = plt.subplots(figsize=(15,10))

    sns.boxenplot(x="TeamName", y='sc', data=tds, ax=axbp)
    ascores=infopd[~infopd['Slot'].isin([20,21])].groupby('TeamName').sum()
    sns.stripplot(data=ascores.reset_index(), x="TeamName", y="Score",color='r',jitter=0,size=10)
    axbp.tick_params(axis='x', labelrotation=45)
    axbp.set_title('Team Possibile Scores Week {}'.format(w));
    axbp.set(xlabel='Team', ylabel='Possible Score');
    fname='data/w{}_overall_prob.png'.format(w)
    plt.savefig(fname)
    return fname

#Create all the projections for it
def gen_projections(sc_data,ffmap,w):
    gnum=1
    proj_scores={}
    games={}
    diffs={}
    all_ros_p=pd.DataFrame([])
    for game in sc_data['schedule']:
        if game['matchupPeriodId']==w:
            home=game['home']
            home_n=ffmap.team_map.get(game['home']['teamId'])
            home_div=ffmap.team_div.get(game['home']['teamId'])
            home_rec=ffmap.rtxt_dict.get(game['home']['teamId'])
            away=game['away']
            away_n=ffmap.team_map.get(game['away']['teamId'])
            away_div=ffmap.team_div.get(game['away']['teamId'])
            away_rec=ffmap.rtxt_dict.get(game['away']['teamId'])

            columns=['Player','AScore','PScore','Slot','Pos','PSlots']

            plines=[]
            for ent in home['rosterForCurrentScoringPeriod']['entries']:
                a_score=ent['playerPoolEntry']['appliedStatTotal']
                for x in ent['playerPoolEntry']['player']['stats']:
                    if x['statSourceId']==1:
                        pl_proj=x
                        p_score=pl_proj['appliedTotal']
                        break
                pline=[ent['playerPoolEntry']['player']['fullName'],
                 a_score,
                 p_score,
                 ent['lineupSlotId'],
                 ffmap.pos_dict.get(ent['lineupSlotId'],'Unknown'),
                 ent['playerPoolEntry']['player']['eligibleSlots']]
                plines.append(pline)
            tlp=pd.DataFrame(plines,columns=columns)
            tlp.insert(column='tid',loc=0,value=home['teamId'])
            tlp.insert(column='gid',loc=0,value=gnum)
            home_pscore=tlp[~tlp['Slot'].isin([20,21])].PScore.sum()
            all_ros_p=all_ros_p.append(tlp)

            plines=[]
            for ent in away['rosterForCurrentScoringPeriod']['entries']:
                a_score=ent['playerPoolEntry']['appliedStatTotal']
                for x in ent['playerPoolEntry']['player']['stats']:
                    if x['statSourceId']==1:
                        pl_proj=x
                        p_score=pl_proj['appliedTotal']
                        break
                pline=[ent['playerPoolEntry']['player']['fullName'],
                 a_score,
                 p_score,
                 ent['lineupSlotId'],
                 ffmap.pos_dict.get(ent['lineupSlotId'],'Unknown'),
                 ent['playerPoolEntry']['player']['eligibleSlots']]
                plines.append(pline)
            tlp=pd.DataFrame(plines,columns=columns)
            tlp.insert(column='tid',loc=0,value=away['teamId'])
            tlp.insert(column='gid',loc=0,value=gnum)
            away_pscore=tlp[~tlp['Slot'].isin([20,21])].PScore.sum()

            all_ros_p=all_ros_p.append(tlp)

            proj_scores[home_n]=home_pscore
            proj_scores[away_n]=away_pscore
            #print(heading)
            gnum=gnum+1
            #print(tlp)
    return proj_scores

#Create the post superlatives and summary table
def get_post_details(infopd, tds, proj_scores):
    ascores=infopd[~infopd['Slot'].isin([20,21])].groupby('TeamName').sum()
    bsc=tds.groupby('TeamName').max()
    stbl=ascores[['Score']].join(bsc)
    stbl.columns=['Actual Score','Best Possible Score']

    bmax=stbl.idxmax().to_frame()
    bmin=stbl.idxmin().to_frame()

    perf=[]
    for gname, gp in infopd[['Name','Score','SlotPos']][~infopd['SlotPos'].isin(['Unknown','BNCK'])].groupby('SlotPos'):
        sc=StandardScaler()
        gp=gp.set_index('Name')
        xs=sc.fit_transform(gp['Score'].values.reshape(-1, 1))
        gp['Scale']=xs
        pmin=gp.loc[gp.Score.idxmin()]
        pmax=gp.loc[gp.Score.idxmax()]
        perf.append([pmax.name,pmin.name,pmax.Scale,pmin.Scale])
    perfdf=pd.DataFrame(perf,columns=['Max_Name','Min_Name','MaxN','MinN'])
    LVP=perfdf.loc[perfdf.MinN.idxmin].Min_Name
    MVP=perfdf.loc[perfdf.MaxN.idxmax].Max_Name

    x = stbl.transpose().values #returns a numpy array
    max_scaler =MaxAbsScaler()
    x_scaled = max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled,columns=bsc.transpose().columns)
    res=df.transpose()
    res['D']=res[1]-res[0]
    worst_manager=res.loc[res.D.idxmax].name
    best_manager=res.loc[res.D.idxmin].name

    stbl['Projected Score']=stbl.index.map(proj_scores)
    scaler = MaxAbsScaler()
    x = stbl[['Projected Score']].transpose().values
    scaler.fit(x)
    xsc2=scaler.transform(stbl[['Actual Score']].transpose().values)
    stbl['Scale_Over_Projection']=list(xsc2.flatten())

    stbl['Projected Score']=stbl['Projected Score'].apply(lambda x: "{:.2f}".format(x))

    Rudy=bsc.loc[stbl['Scale_Over_Projection'].idxmax].name
    MH=bsc.loc[stbl['Scale_Over_Projection'].idxmin].name
    stbl.index.name=''
    post={'tbl':stbl[['Actual Score','Best Possible Score','Projected Score']].to_markdown(),
         'HS':bmax.loc['Actual Score'][0],
         'HBS':bmax.loc['Best Possible Score'][0],
         'LS':bmin.loc['Actual Score'][0],
         'LBS':bmin.loc['Best Possible Score'][0],
         'MVP':MVP,
         'LVP':LVP,
         'WM':worst_manager,
         'BM':best_manager,
         'Rudy':Rudy,
         'MH':MH}

    return post

def upload_game_images(gplots):
    # Admin API key goes here
    key = '5f808148c9c9380001c19663:940f1101a860af76263129344d8b98576d0070277f827fe27f1db34f44b223b3'

    # Split the key into ID and SECRET
    id, secret = key.split(':')

    # Prepare header and payload
    iat = int(date.now().timestamp())

    header = {'alg': 'HS256', 'typ': 'JWT', 'kid': id}
    payload = {
        'iat': iat,
        'exp': iat + 5 * 60,
        'aud': '/v3/admin/'
    }

    # Create the token (including decoding secret)
    token = jwt.encode(payload, bytes.fromhex(secret), algorithm='HS256', headers=header)

    # Make an authenticated request to create a post
    url_img = 'http://localhost:3001/ghost/api/v3/admin/images/upload'
    url_post = 'http://localhost:3001/ghost/api/v3/admin/posts/'
    headers = {'Authorization': 'Ghost {}'.format(token.decode())}

    for i,g in gplots.items():
        imloc=g['impath']
        files={'file':(imloc,
                   open(imloc,'rb'),
                   'image/png',{'resource':'image'})}
        r=requests.post(url_img,files=files,headers=headers)
        if r.status_code==201:
            new_path=r.json()['images'][0]['url']
            g.update({'img':new_path})
        else:
            raise ConnectionError
    return gplots

def upload_overall_image(path):
    # Admin API key goes here
    key = '5f808148c9c9380001c19663:940f1101a860af76263129344d8b98576d0070277f827fe27f1db34f44b223b3'

    # Split the key into ID and SECRET
    id, secret = key.split(':')

    # Prepare header and payload
    iat = int(date.now().timestamp())

    header = {'alg': 'HS256', 'typ': 'JWT', 'kid': id}
    payload = {
        'iat': iat,
        'exp': iat + 5 * 60,
        'aud': '/v3/admin/'
    }

    # Create the token (including decoding secret)
    token = jwt.encode(payload, bytes.fromhex(secret), algorithm='HS256', headers=header)

    # Make an authenticated request to create a post
    url_img = 'http://localhost:3001/ghost/api/v3/admin/images/upload'
    url_post = 'http://localhost:3001/ghost/api/v3/admin/posts/'
    headers = {'Authorization': 'Ghost {}'.format(token.decode())}

    imloc=path
    files={'file':(imloc,
                       open(imloc,'rb'),
                       'image/png',{'resource':'image'})}
    r=requests.post(url_img,files=files,headers=headers)
    sum_path=r.json()['images'][0]['url']
    return sum_path

#--------------------------------------------------------------------------#

def espn_data_download():
    #w=4
    lstart=Variable.get("NFL_START_DATE")
    w=datetime.today().isocalendar()[1]-datetime.strptime('2020-9-11','%Y-%m-%d').isocalendar()[1]
    Variable.set("week",str(w))
    #lid=866268
    lid=Variable.get("ESPN_LEAGUE")
    sc_data=get_scoreboard(w,lid)
    with open('data/sc_data_{}.json'.format(w), 'w') as outfile:
        json.dump(sc_data, outfile)

def data_processing():
    w=4
    with open('data/sc_data_{}.json'.format(w)) as f:
        sc_data = json.load(f)

    ffmap=ffantasy_maps(sc_data)

    infopd,summary=create_week_data(sc_data, ffmap, w)

    pos_scores=get_all_pos_scores(infopd, ffmap)

    headings=get_headings(summary,ffmap)

    with open('data/headings_data_{}.json'.format(w), 'w') as outfile:
        json.dump(headings, outfile)

    gplots=gen_game_plots(summary,pos_scores,w)

    with open('data/plots_data_{}.json'.format(w), 'w') as outfile:
        json.dump(gplots, outfile)

    tds=gen_tds(pos_scores)

    ofig_path=gen_overall_chart(tds,infopd,w)

    proj_scores=gen_projections(sc_data,ffmap,w)

    post=get_post_details(infopd, tds, proj_scores)

    with open('data/post_{}.json'.format(w), 'w') as outfile:
        json.dump(post, outfile)

def set_token():
    key=Variable.get("GHOST_SECRET")
    # Split the key into ID and SECRET
    id, secret = key.split(':')

    # Prepare header and payload
    iat = int(date.now().timestamp())

    header = {'alg': 'HS256', 'typ': 'JWT', 'kid': id}
    payload = {
        'iat': iat,
        'exp': iat + 5 * 60,
        'aud': '/v3/admin/'
    }

    # Create the token (including decoding secret)
    token = jwt.encode(payload, bytes.fromhex(secret), algorithm='HS256', headers=header)

    Variable.set("GHOST_TOKEN_SECRET", token)

def data_upload():
    w=4
    with open('data/plots_data_{}.json'.format(w)) as f:
        gplots = json.load(f)
    opath='data/w{}_overall_prob.png'.format(w)
    ugplots=upload_game_images(gplots)
    uopath=upload_overall_image(opath)

    ugplots.update({'o':uopath})

    with open('data/upplots_data_{}.json'.format(w), 'w') as outfile:
        json.dump(ugplots, outfile)

def gen_mobiledoc():
    w=4
    with open('data/upplots_data_{}.json'.format(w)) as f:
        upplots = json.load(f)

    with open('data/post_{}.json'.format(w)) as f:
        posts = json.load(f)

    with open('data/headings_data_4.json'.format(w)) as f:
        headings = json.load(f)

    top_heading= \
    '> some quote  \n  \n' +\
    'Team with the Highest Score: {}  \n'.format(posts['MH']) +\
    'Team with the Highest Best Score: {}  \n'.format(posts['HBS']) +\
    'Team with the Lowest Score: {}  \n'.format(posts['LS']) +\
    'Team with the Lowest Best Score: {}  \n'.format(posts['LBS']) +\
    posts['tbl'] +\
    '   \n'+ \
    'Week {} MVP: {}  \n'.format(w,posts['MVP']) +\
    'Week {} LVP: {}  \n'.format(w, posts['LVP']) +\
    '## Awards  \n' +\
    "Worst Manager: {}  \n".format('WM') +\
    "Future Seer: {}  \n".format('BM') +\
    "Rudy: {}  \n".format(posts['Rudy'])+\
    "Most Harmless: {}  \n".format(posts['MH'])


    cards=[]
    cards.append(['markdown',{'cardName':'markdown','markdown':top_heading}])
    i=1
    for i in range(1,len(headings)+1):
        cards.append(['markdown',{'cardName':'markdown','markdown':'### '+headings[str(i)]}])
        cards.append(['image',{'cardName':'Image','src':upplots[str(i)]['img']}])

    cards.append(['markdown',{'cardName':'markdown','markdown':'## Overal Performance'}])
    cards.append(['image',{'cardName':'Overal Image','src':upplots['o']}])

    sections=[[10,x] for x in range(0,len(cards))]

    mobiledoc = json.dumps({
        'version': '0.3.1',
        'markups': [],
        'atoms': [],
        'cards': cards,
        'sections': sections
    });

    with open('data/mobile_{}.json'.format(w), 'w') as outfile:
        json.dump(mobiledoc, outfile)

def upload_post():
    w=4
    key = '5f808148c9c9380001c19663:940f1101a860af76263129344d8b98576d0070277f827fe27f1db34f44b223b3'

    # Split the key into ID and SECRET
    id, secret = key.split(':')

    # Prepare header and payload
    iat = int(date.now().timestamp())

    header = {'alg': 'HS256', 'typ': 'JWT', 'kid': id}
    payload = {
        'iat': iat,
        'exp': iat + 5 * 60,
        'aud': '/v3/admin/'
    }

    # Create the token (including decoding secret)
    token = jwt.encode(payload, bytes.fromhex(secret), algorithm='HS256', headers=header)

    with open('data/mobile_{}.json'.format(w)) as f:
        mobiledoc = json.load(f)

    url_post = 'http://localhost:3001/ghost/api/v3/admin/posts/'
    headers = {'Authorization': 'Ghost {}'.format(token.decode())}
    body = {'posts': [{'title': 'Week {} Recap'.format(w),
                  'mobiledoc':mobiledoc}]}
    r = requests.post(url_post, json=body, headers=headers)
#----------------------------------------------------------------------------#
# TEST CALLS
"""
espn_data_download()
data_processing()
data_upload()
gen_mobiledoc()
upload_post()
"""
#-----------------------------------------------------------------------------#


with DAG("ESPNBLOG", start_date=days_ago(2), default_args=default_args, catchup=False, schedule_interval = "@weekly",) as dag:
    fetch_espn_data = PythonOperator(task_id="espn_data_download",python_callable=espn_data_download,provide_context=True)


    #fetchDataToLocal = PythonOperator(task_id="fl_load",python_callable=fetchDataToLocal,provide_context=True)

    #sqlLoad = PythonOperator(task_id="sql_load", python_callable=load2PS, provide_context=True)
    fetch_espn_data
    #fetchDataToLocal >> sqlLoad
