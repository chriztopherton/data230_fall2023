import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.ensemble import IsolationForest
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")


#------------------------------------------------------------------------------------------------------------------------------------------------
# Incorporate data
@st.cache_data
def load_data():
    data = pd.read_csv("Alternative_Fueling_Stations.csv",low_memory=False)\
            .drop(['groups_with_access_code_fr','id'],axis=1)
    data['open_date'] = pd.to_datetime(data['open_date'])
    data['open_date_date'] = pd.to_datetime(data['open_date'].dt.date)
    data['open_date_year'] = data['open_date'].dt.year
    data['MonthNum'] = data['open_date'].dt.month
    data['MonthName'] = data['open_date'].dt.month_name()
    data['MonthNameShort'] = data['open_date'].dt.month_name().str[:3]

    data['Weekday'] = data['open_date'].dt.weekday
    data['WeekName'] = data['open_date'].dt.day_name()
    data['DayNameShort'] = data['open_date'].dt.day_name().str[:3]
    data['DayOfMonth'] = data['open_date'].dt.day
    data['DayOfYear'] = data['open_date'].dt.dayofyear

    data['QuarterOfYear'] = data['open_date'].dt.quarter

    return data

display_cols = ['station_name','station_phone','street_address','access_code','access_days_time',
              'city','date_last_confirmed','fuel_type_code','groups_with_access_code']

@st.cache_data
def groupby_data(data):
    state_city_stations = data.groupby(['city'])[['latitude','longitude']].median().reset_index()\
        .merge(data.groupby(['city']).size().to_frame("count_stations"), left_on = "city",right_on="city")\
        .sort_values('count_stations',ascending=False).reset_index().drop('index',axis=1)
    state_city_stations['city'] = [i.title() for i in state_city_stations['city']]
    state_city_stations['text'] = [str(i) + '<br>stations ' for i in state_city_stations['count_stations'] ]

    return state_city_stations


@st.cache_data
def display_map(d,col):
    fig = px.scatter_mapbox(d, 
                    lat="latitude", 
                    lon="longitude", 
                    title="Map - px.scatter_mapbox",
                    color=col,
                    hover_data=["ev_network","station_name","station_phone","fuel_type_code",'ev_connector_types','open_date_date'],
                    zoom = 3, size_max=15)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig

@st.cache_data
def display_map_station_counts(d):

    limits = [(1,2),(3,220),(221,10636)]
    colors = ["royalblue","crimson","lightseagreen"]
    cities = []
    scale = 30

    fig_stations_count = go.Figure()

    for i in range(len(limits)):
        lim = limits[i]
        df_sub = d[lim[0]:lim[1]]
        fig_stations_count.add_trace(go.Scattergeo(
            locationmode = 'USA-states',
            lon = df_sub['longitude'],
            lat = df_sub['latitude'],
            text = df_sub['text'],
            marker = dict(
                size = df_sub['count_stations']/scale,
                color = colors[i],
                line_color='rgb(40,40,40)',
                line_width=0.5,
                sizemode = 'area'
            ),
            name = '{0} - {1}'.format(lim[0],lim[1])))

    fig_stations_count.update_layout(
            title_text = '2023 US city charging stations <br>(Click legend to toggle traces)',
            showlegend = True,
            geo = dict(
                scope = 'usa',
                landcolor = 'rgb(217, 217, 217)',
            )
        )
    
    return fig_stations_count


def main():

    data = load_data()

    st.sidebar.title("Alternative Fueling Stations")

    min_date = data['open_date_date'].min().date()
    max_date = data['open_date_date'].max().date()
    date_range = st.sidebar.slider('Select date', 
                        min_value=min_date, 
                        value=(min_date, max_date),
                        max_value=max_date, format="MM/DD/YYYY")


    mask = (data['open_date_date'] > pd.to_datetime(date_range[0])) & (data['open_date_date'] <= pd.to_datetime(date_range[1]))
    d = data.loc[mask]

    #------------------------------

    wc_feat = st.sidebar.selectbox("Choose attributes to build wordcloud",['station_name','state','city','access_days_time','groups_with_access_code'])

    text = d[[wc_feat]].dropna()[wc_feat].values.tolist()
    wordcloud = WordCloud().generate(' '.join(text))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.show()
    st.sidebar.pyplot()

    #------------------------------
    st.sidebar.caption(f"There were {d.shape[0]} stations opened between {date_range[0]} and {date_range[1]}")
    st.sidebar.dataframe(d[display_cols],hide_index=True) 

    col1,col2 = st.columns([0.4,0.6])

    with col1:
        color_col = st.selectbox("Color by:", ['fuel_type_code','access_code',"groups_with_access_code"])
        pie_df = data.groupby(color_col).size().to_frame('size').reset_index()

        fig_pie = px.pie(pie_df, values="size", names = color_col,
                        height=300, width=300)
        fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
        st.plotly_chart(fig_pie)
    with col2:
        st.plotly_chart(display_map(d,color_col), use_container_width=True)

    state_city_stations = groupby_data(data)

    #tab1 = st.tabs(["Map - pdk.Deck"])

    # with tab1:
    #     st.plotly_chart(display_map_station_counts(state_city_stations),use_container_width=True,theme="streamlit")

    #with tab1:

    chart_data = state_city_stations.rename(columns={'latitude':'lat','longitude':'lon'})
    scaler = MinMaxScaler()
    chart_data[['count_stations_scaled']] = scaler.fit_transform(chart_data[['count_stations']])

    st.subheader("Pydeck")
    st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=34.059422,
        longitude=-118.333892,
        zoom=8,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
        'HexagonLayer',
        data=chart_data,
        get_position='[lon, lat]',
        radius=200,
        elevation_scale=4,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],
),use_container_width=True)
        

    bar1,bar2 = st.tabs(["Bar - Single Variables", "Bar - Grouped Variables"])

    with bar1:

        #with col2:
        col1,col2 = st.columns([0.2,0.8])
        var_value_counts = col1.selectbox("Select feature to summarize",
                                                ['fuel_type_code','access_code','groups_with_access_code','status_code','geocode_status',
                                                'state','ev_connector_types','ev_network', 'ev_network_web'],
                                                placeholder="Choose an attribute")
        var_value_counts_df = d[var_value_counts].value_counts().to_frame("size").reset_index().sort_values('size',ascending=True)
        fig = px.histogram(var_value_counts_df, y=var_value_counts, x='size')
        col2.plotly_chart(fig,theme="streamlit",use_container_width=True)

    with bar2:
        col1, col2 = st.columns(2)
        gp1 = col1.selectbox("",
                                                ['fuel_type_code','access_code','groups_with_access_code','status_code','geocode_status',
                                                'state','ev_connector_types','ev_network', 'ev_network_web'],
                                                placeholder="Choose a first-level variable")
        gp2 = col2.selectbox("",
                                                [i for i in ['fuel_type_code','access_code','groups_with_access_code','status_code','geocode_status',
                                                'state','ev_connector_types','ev_network', 'ev_network_web'] if i != gp1],
                                                placeholder="Choose a second-level variable")
        
        gp_nested = d.groupby([gp1])[gp2].value_counts().to_frame('size').reset_index().sort_values('size',ascending=True)
        fig_gp = px.histogram(gp_nested, y=gp1, x='size',color= gp2)
        st.plotly_chart(fig_gp,theme="streamlit",use_container_width=True)

        #state_city = st.selectbox("Choose to either visualize by state or city:",['state','city'])

    line1,line2 = st.tabs(["Line - Yearly Cummulative Stations Growth by States","Line - Electric Charging Growth"])

    with line1:
        df_csum = d.groupby(['open_date_year','state','city']).size().to_frame('size')['size'].cumsum().reset_index()
        st.plotly_chart(px.line(df_csum, x = df_csum["open_date_year"], y=df_csum['size'],color='state',
                                title=f'Growth of charging stations between {date_range[0]} and {date_range[1]}'),theme="streamlit",use_container_width=True)
        
    with line2:
        elec = d.query('fuel_type_code == "ELEC"').sort_values('open_date')
        elec_charge_df = elec.groupby(elec['open_date'].dt.to_period('Q'))[['ev_dc_fast_num','ev_level1_evse_num', 'ev_level2_evse_num']]\
            .sum().reset_index()
        elec_charge_df_melt = elec_charge_df.melt(id_vars='open_date', value_vars=['ev_dc_fast_num','ev_level1_evse_num', 'ev_level2_evse_num'])
        st.plotly_chart(px.line(elec_charge_df_melt, x = elec_charge_df_melt["open_date"].dt.strftime("%Y-%m"), y=elec_charge_df_melt['value'],color='variable',
                                title=f'Quarterly Growth of EV stations between {date_range[0]} and {date_range[1]}'),theme="streamlit",use_container_width=True)
        
        #------------------------------------------------------------------------------------------
        st.subheader("ML Analysis with Isolation Forest to identify anomalies")
        iso_df = elec_charge_df.set_index('open_date')
    



        col1,col2 = st.columns([0.65,0.35])
        with col1:
            contam = st.slider("Select contamination parameter for IF model",min_value=0.0,max_value=0.5,value = 0.1)
            model_IF = IsolationForest(contamination=float(contam),random_state=42)
            model_IF.fit(iso_df)
            iso_df['anomaly_scores'] = model_IF.decision_function(iso_df)
            iso_df['anomaly'] = model_IF.predict(iso_df.drop('anomaly_scores',axis=1))
            iso_df.index = [i.strftime("%Y-%m") for i in iso_df.index]

            anomaly_dates = iso_df.sort_values("anomaly_scores").query('anomaly == -1')
            non_anomaly_dates = iso_df.sort_values("anomaly_scores").query('anomaly == 1')

            st.dataframe(anomaly_dates)

        with col2:

            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.scatter(anomaly_dates['ev_dc_fast_num'],
                            anomaly_dates['ev_level1_evse_num'],
                            anomaly_dates['ev_level2_evse_num'])
            ax.scatter(non_anomaly_dates['ev_dc_fast_num'],
                                non_anomaly_dates['ev_level1_evse_num'],
                                non_anomaly_dates['ev_level2_evse_num'])
            ax.set_xlabel('ev_dc_fast_num')
            ax.set_ylabel('ev_level1_evse_num')
            ax.set_zlabel('ev_level2_evse_num')
            st.pyplot(fig)
    




        




#------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
