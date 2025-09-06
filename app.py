import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os



# Import main df
df = pd.read_csv(rf"data/cc_clean.csv")
final_account_labels = pd.read_csv(rf"data/final_acct_table.csv")

# Figure out which columns to bring from final_account_labels
# Always include acct_num2 and cluster when merging
cols_to_add = [c for c in final_account_labels.columns if c not in df.columns or c == "acct_num2"]

# Merge cleanly
df1 = df.merge(final_account_labels[cols_to_add], on="acct_num2", how="left")



# Convert and add columns

df1['trans_datetime'] = pd.to_datetime(df1['trans_datetime'])
df1['trans_month_year'] = df1['trans_datetime'].dt.to_period('M')
df1['YEAR'] = df1['trans_datetime'].dt.to_period('Y')
df1['category'] = df1['category'].fillna('Unknown')

# Drop duplicate txns potentially from joins
df1 = df1.drop_duplicates(subset=['trans_num'])



st.set_page_config(page_title="My first streamlit project",layout="wide")

# Sidebar
st.sidebar.title("Resilient VS Rebound")
menu = st.sidebar.radio("",["Overview","Resilient Essentials","Rebound Discretionary","Dormant Big-Ticket"])

data_dir = "data"
image_dir = "images"
plot_dir = "plots"

if menu == "Overview": 
    st.markdown("# Premise")
    st.markdown("Our project is based on Adobo Bank’s credit card transaction data from 2020 to 2021 — the height of the pandemic. The challenge we set out to solve is simple but important: How can we segment customers into meaningful groups, so the bank can tailor its acquisition, retention, and campaign strategies?")
    

    st.markdown("# Hypothesis: Two Broad Behaviors")
    st.markdown("we hypothesize that customers exhibit two broad behaviors: Resilient Essentials — those who stayed focused on necessities, and Rebound Discretionary — those who bounced back to non-essential categories.")
    overall_image_path = os.path.join(image_dir, "hypothesis.png")
    st.image(overall_image_path, caption="", use_container_width=True)

    st.markdown("# Original Dataset")
    st.markdown("The dataset from Adobo Bank includes transactions of its cardholders from 2020-2021")
    st.dataframe(df.head(20))


    # Download button for merged dataset
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv,
        file_name="cleaned_transactions.csv",
        mime="text/csv",
        key="download_cleaned"
    )

    st.markdown("# Data Preprocessing Pipeline")
    overall_image_path = os.path.join(image_dir, "preprocessing.png")
    st.image(overall_image_path, caption="", use_container_width=True)

    st.markdown("# EDA")
    overall_image_path = os.path.join(image_dir, "eda highlights 1.png")
    st.image(overall_image_path, caption="", use_container_width=True)

    overall_image_path = os.path.join(image_dir, "eda highlights 2.png")
    st.image(overall_image_path, caption="", use_container_width=True)

    st.markdown("# Features")
    st.markdown("To capture these patterns, we engineered three groups of features")
    overall_image_path = os.path.join(image_dir, "features.png")
    st.image(overall_image_path, caption="", use_container_width=True)

    st.markdown("# Final Table")
    st.markdown("Final features attached to each accountnumber to be used in clustering")
    st.dataframe(final_account_labels.head(20))

        # Download button for merged dataset
    csv2 = df1.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv2,
        file_name="final_table.csv",
        mime="text/csv",
        key="download_final"
    )


    st.markdown("# Method & Validation (K-Means)")
    st.markdown("We used K-Means clustering, after scaling the features so that all variables carried equal weight. To determine the optimal number of clusters, we ran elbow and silhouette tests — both pointed to three as the best balance.This gave us three distinct clusters of customers, which we can now profile in detail")
    overall_image_path = os.path.join(image_dir, "k-means.png")
    st.image(overall_image_path, caption="", use_container_width=True)

    st.markdown("# Cluster Behaviors")
    st.markdown("## Recency,Frequency and Monetary Comparison")
    st.markdown("The Transactions per Active Month chart shows how frequently each group spends. Cluster 1 stands out as very active, with the highest transaction counts, while Cluster 2 transacts only rarely.Looking at the Average Monthly Spend chart in the middle, we see that Cluster 1 also leads in total spend, Cluster 0 maintains moderate steady spend, and Cluster 2 spends less frequently but still at substantial levels.Now, on the right, the Median Transaction Amount chart highlights what makes Cluster 2 unique — when they do spend, they make very large-ticket purchases compared to the smaller, steadier amounts of Clusters 0 and Cluster 1")
    overall_image_path = os.path.join(image_dir, "rfm cluster behavior.png")
    st.image(overall_image_path, caption="", use_container_width=True)

    st.markdown("## Category Comparison")
    st.markdown("The Spending Mix by Cluster chart ties this all together. Cluster 0 keeps a balanced mix of categories, Cluster 1 allocates more toward shopping and travel, and Cluster 2 concentrates their spend heavily in shopping and miscellaneous categories.So overall: Cluster 0 are steady, balanced spenders, Cluster 1 are affluent, high-frequency shoppers, and Cluster 2 are infrequent but high-ticket spenders.With these results, let’s translate the clusters into clear business-friendly personas — and tie them back to our Resilient vs Rebound framing")
    overall_image_path = os.path.join(image_dir, "cluster category.png")
    st.image(overall_image_path, caption="", use_container_width=True)
    overall_image_path = os.path.join(image_dir, "cluster category 2.png")
    st.image(overall_image_path, caption="", use_container_width=True)

    st.markdown("# Personas")
    st.markdown("These clusters map back to our Resilient vs Rebound framing. Cluster 0, the Older Urban Steady Spenders, represent Resilient Essentials — cautious, consistent customers who stuck with groceries and daily needs. Cluster 1, Affluent Older Shoppers, are our Rebound Discretionary — bouncing back strongly in shopping and travel. Cluster 2 are Dormant Big-Ticket Shoppers — low activity overall, but with big purchases when they spend.With these personas in hand, we can now connect them to concrete marketing strategies tailored to stability on one side, and growth on the other. ")
    overall_image_path = os.path.join(image_dir, "personas.png")
    st.image(overall_image_path, caption="", use_container_width=True)




elif menu == "Resilient Essentials": 
    st.markdown("#Resilient Essentials Transaction Profile")
    st.markdown("## When did Resilient Essentials customers last transact?")

    recency_ranges = ['(0, 30]', '(30, 60]', '(60, 90]']
    number_of_accounts = [40, 0, 6]

    plt.figure(figsize=(10, 5))
    plt.bar(recency_ranges, number_of_accounts, color='#603470', edgecolor='black')

    plt.title('Resilient Essentials Transaction Recency Distribution')
    plt.xlabel('Recency Range (days)')
    plt.ylabel('Number of Accounts')

    for i, val in enumerate(number_of_accounts):
        plt.text(i, val + 0.5, str(val), ha='center')

    plt.tight_layout()
    st.pyplot(plt)   # ✅ no plt.show()

    st.markdown("## How often do Resilient Essentials customers transact?")
    # Cluster = 0
    cluster = 0
    aggregate= df1[['trans_month_year','trans_num','acct_num2']][df1['cluster']==cluster].groupby(['trans_month_year','acct_num2']).count().reset_index()
    aggregate1 = aggregate[['trans_month_year','trans_num']].groupby(['trans_month_year']).mean().reset_index() 
    aggregate1

    # Convert period[M] to timestamp if needed
    aggregate1['trans_month_year'] = aggregate1['trans_month_year'].dt.to_timestamp()

    # Extract year and month
    aggregate1['year'] = aggregate1['trans_month_year'].dt.year
    aggregate1['month'] = aggregate1['trans_month_year'].dt.month

    # Group by year and month (already aggregated but keeping structure consistent)
    grouped = aggregate1.groupby(['year', 'month'], as_index=False)['trans_num'].mean()

    # Pivot for plotting
    pivot_df = grouped.pivot(index='month', columns='year', values='trans_num')

    # Calculate median per year
    medians = grouped.groupby('year')['trans_num'].median()

    # Plot
    plt.figure(figsize=(10, 6))

    # Monthly transactions (solid pastel blue)
    if 2020 in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[2020], label='2020 Monthly', color='#603470', alpha=0.2)
    if 2021 in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[2021], label='2021 Monthly', color='#603470', alpha=1.0)

    # Median transactions (horizontal broken pastel green)
    if 2020 in medians.index:
        plt.hlines(medians[2020], xmin=1, xmax=12, colors='lightgreen', linestyles='--', label='2020 Median', alpha=0.4)
    if 2021 in medians.index:
        plt.hlines(medians[2021], xmin=1, xmax=12, colors='lightgreen', linestyles='--', label='2021 Median', alpha=1.0)

    # Customize
    plt.title("Resilient Essentials Transactions per Month")
    plt.xlabel('Month')
    plt.ylabel('Avg Transactions per User')
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    plt.show()
    st.pyplot(plt)   # ✅ no plt.show()

elif menu == "Rebound Discretionary": 
    st.markdown("#Rebound Discretionary Transaction Profile")
    st.markdown("## When did Rebound Discretionary customers last transact?")

    recency_ranges = ['(0, 30]', '(30, 60]']
    number_of_accounts = [24, 2]

    plt.figure(figsize=(10, 5))
    plt.bar(recency_ranges, number_of_accounts, color='#d25784', edgecolor='black')

    plt.title('Rebound Discretionary Transaction Recency Distribution')
    plt.xlabel('Recency Range (days)')
    plt.ylabel('Number of Accounts')

    for i, val in enumerate(number_of_accounts):
        plt.text(i, val + 0.5, str(val), ha='center')

    plt.tight_layout()
    st.pyplot(plt)   # ✅ no plt.show()




    st.markdown("## How often do Rebound Discretionary customers transact?")
    # Cluster = 1
    cluster = 1
    aggregate= df1[['trans_month_year','trans_num','acct_num2']][df1['cluster']==cluster].groupby(['trans_month_year','acct_num2']).count().reset_index()
    aggregate1 = aggregate[['trans_month_year','trans_num']].groupby(['trans_month_year']).mean().reset_index() 
    aggregate1

    # Convert period[M] to timestamp if needed
    aggregate1['trans_month_year'] = aggregate1['trans_month_year'].dt.to_timestamp()

    # Extract year and month
    aggregate1['year'] = aggregate1['trans_month_year'].dt.year
    aggregate1['month'] = aggregate1['trans_month_year'].dt.month

    # Group by year and month (already aggregated but keeping structure consistent)
    grouped = aggregate1.groupby(['year', 'month'], as_index=False)['trans_num'].mean()

    # Pivot for plotting
    pivot_df = grouped.pivot(index='month', columns='year', values='trans_num')

    # Calculate median per year
    medians = grouped.groupby('year')['trans_num'].median()

    # Plot
    plt.figure(figsize=(10, 6))

    # Monthly transactions (solid pastel blue)
    if 2020 in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[2020], label='2020 Monthly', color='#d25784', alpha=0.4)
    if 2021 in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[2021], label='2021 Monthly', color='#d25784', alpha=1.0)

    # Median transactions (horizontal broken pastel green)
    if 2020 in medians.index:
        plt.hlines(medians[2020], xmin=1, xmax=12, colors='lightgreen', linestyles='--', label='2020 Median', alpha=0.2)
    if 2021 in medians.index:
        plt.hlines(medians[2021], xmin=1, xmax=12, colors='lightgreen', linestyles='--', label='2021 Median', alpha=1.0)

    # Customize
    plt.title("Rebound Discretionary Transactions per Month")
    plt.xlabel('Month')
    plt.ylabel('Avg Transactions per User')
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    st.pyplot(plt)   # ✅ no plt.show()

elif menu == "Dormant Big-Ticket": 
    st.markdown("#Dormant Big-Ticket Transaction Profile")
    st.markdown("## When did Dormant Big-Ticket customers last transact??")

    recency_bins = [
        '(110, 140]', '(140, 170]', '(170, 200]', '(200, 230]', '(230, 260]',
        '(260, 290]', '(290, 320]', '(320, 350]', '(350, 380]', '(380, 410]',
        '(410, 440]', '(440, 470]', '(470, 500]', '(500, 530]', '(530, 560]',
        '(560, 590]', '(590, 620]'
    ]

    account_counts = [2, 2, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    plt.figure(figsize=(12, 6))
    plt.bar(recency_bins, account_counts, color='#e6a752', edgecolor='black')
    plt.xticks(rotation=45, ha='right')

    plt.title('Dormant Big-Ticket Transaction Recency Distribution')
    plt.xlabel('Recency Range (days)')
    plt.ylabel('Number of Accounts')

    for i, val in enumerate(account_counts):
        if val > 0:
            plt.text(i, val + 0.2, str(val), ha='center')

    plt.tight_layout()
    st.pyplot(plt)   # ✅ no plt.show()

    st.markdown("## How often do Dormant Big-Ticket customers transact?")

    # Cluster = 2
    cluster = 2
    aggregate= df1[['trans_month_year','trans_num','acct_num2']][df1['cluster']==cluster].groupby(['trans_month_year','acct_num2']).count().reset_index()
    aggregate1 = aggregate[['trans_month_year','trans_num']].groupby(['trans_month_year']).mean().reset_index() 
    aggregate1

    # Convert period[M] to timestamp if needed
    aggregate1['trans_month_year'] = aggregate1['trans_month_year'].dt.to_timestamp()

    # Extract year and month
    aggregate1['year'] = aggregate1['trans_month_year'].dt.year
    aggregate1['month'] = aggregate1['trans_month_year'].dt.month

    # Group by year and month (already aggregated but keeping structure consistent)
    grouped = aggregate1.groupby(['year', 'month'], as_index=False)['trans_num'].mean()

    # Pivot for plotting
    pivot_df = grouped.pivot(index='month', columns='year', values='trans_num')

    # Calculate median per year
    medians = grouped.groupby('year')['trans_num'].median()

    # Plot
    plt.figure(figsize=(10, 6))

    # Monthly transactions (solid pastel blue)
    if 2020 in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[2020], label='2020 Monthly', color='#e6a752', alpha=0.4)
    if 2021 in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[2021], label='2021 Monthly', color='#e6a752', alpha=1.0)

    # Median transactions (horizontal broken pastel green)
    if 2020 in medians.index:
        plt.hlines(medians[2020], xmin=1, xmax=12, colors='lightgreen', linestyles='--', label='2020 Median', alpha=0.2)
    if 2021 in medians.index:
        plt.hlines(medians[2021], xmin=1, xmax=12, colors='lightgreen', linestyles='--', label='2021 Median', alpha=1.0)

    # Customize
    plt.title("Dormant Big-Ticket Transactions per Month")
    plt.xlabel('Month')
    plt.ylabel('Avg Transactions per User')
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    plt.show()
    st.pyplot(plt)   # ✅ no plt.show()    





