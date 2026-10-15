_QUERY_STOPWORDS = """
a an and are as at be by can do does for from has have how i if in into is it its of on or
that the their them then there these this to was what when where which who why will with you your
"""

_FACT_STOPWORDS = """
a an and are as at be by for from in is it of on or that the this to was with
"""

STOPWORDS = frozenset(_QUERY_STOPWORDS.split())
FACT_STOPWORDS = frozenset(_FACT_STOPWORDS.split())
