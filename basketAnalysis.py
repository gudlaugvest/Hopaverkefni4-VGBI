
import pandas as pd
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# connection credentials
db_server = "vgbi707.database.windows.net"
db_name = "vgbi2025nem"
db_username = "vgbiH17"
db_password = "JfSJapDatzxCCi0s"

# ODBC Driver
odbc_driver = "{ODBC Driver 18 for SQL SERVER}"

# connection strings
connection_string_sqlalchemy = (
"mssql+pyodbc://{username}:{password}@{db_server}/{db_name}"
    "?driver=ODBC+Driver+18+for+SQL+Server"
).format(
    username=db_username,
    password=db_password,
    db_server=db_server,
    db_name=db_name
)

# create a connection to the database
engine = create_engine(connection_string_sqlalchemy)

# construct a query that joins factSales with dimProducts to get product names
query = """
SELECT s.receipt, s.idCalendar, p.name
FROM h17.factSales AS s
LEFT JOIN h17.dimProduct AS p 
  ON s.idProduct = p.sourceId
"""

# read the joined data into a DataFrame
df = pd.read_sql_query(query, engine)
print("DATA WITH PRODUCT NAMES \n", df.head(10))

# create a unique transaction identifier
df['single_transaction'] = df['receipt'].astype(str) + '_' + df['idCalendar'].astype(str)
print("SINGLE TRANSACTION \n", df.head(10))


# create a pivot table
df2 = pd.crosstab(df['single_transaction'], df['name'])
print(df2.head(10))

# encode the data
def encode(item_freq):
    res = 0
    if item_freq > 0:
        res = 1
    return res
    
basket_input = df2 > 0

# apply the apriori algorithm
frequent_itemsets = apriori(basket_input, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift")
print(rules.head(10))

# convert frozensets to comma-separated strings
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))

# write to csv
rules.to_csv("association_rules_clean.csv", index=False)