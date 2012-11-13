"""
Utilities for collaborative filtering
"""
import numpy as np
import pandas as pd

def make_ratings_hash(ratings):
    """Make a hashtable of ratings indexed by (userid,itemid)"""
    rhash = {}
    for row_indx, (userid, itemid, rating) in ratings[['userid','itemid','rating']].iterrows():
        rhash[(userid,itemid)]=rating
    return rhash

def make_item_ratings_hash(ratings):
    """Make a hashtable of ratings indexed by (userid,itemid)"""
    rhash = {}
    for row_indx, (userid, itemid) in ratings[['userid','itemid']].iterrows():
        rhash[(userid,itemid)]=ratings.xs(row_indx)[ 'Action':'isgood' ]
    return rhash
    
def make_item_norms_hash(ratings):
    """Make a hashtable of ratings indexed by (userid,itemid)"""
    rhash = {}
    for row_indx, (userid, itemid) in ratings[['userid','itemid']].iterrows():
        nope = False
        #print row_indx
        
        '''
        for x in ratings.xs(row_indx):
            if isInstance(ratings.xs(row_indx)[x],NoneType):
                nope = True
               
        if nope is not True:
        '''
        rhash[(userid,itemid)]=np.sqrt(ratings.xs(row_indx)[ 'Action':'isgood' ].dot(ratings.xs(row_indx)[ 'Action':'isgood' ]))
            
    return rhash
    
def get_user_neighborhood(ratings, userid, size, norms):
    """
    Find the nearest user's and their cosine similarity to a given user

    Arguments:
    ratings -- a ratings hash table as produced by make_ratings_hash
    userid -- the id of user neighborhood is being calculated
    size -- the number of users in userid's neighborhood
    norms -- a named vector (pandas.Series) of user l2 norms in rating space

    Returns:
    users -- a vector of the ids of the nearest users
    weights -- a vector of cosine similarities for the neighbors
    """
    hash = {}
    for (otheruserid,itemid),rating in ratings.iteritems():
        if otheruserid == userid:
            continue
        if (userid, itemid) not in ratings:
            continue

        if otheruserid not in hash:
            hash[otheruserid] = 0

        hash[otheruserid] += ratings[(userid,itemid)] * rating

    for (otheruserid, val) in hash.iteritems():
        nx=norms[userid]
        ny=norms[otheruserid]
        hash[otheruserid] = hash[otheruserid]/float(nx*ny)

    indx = np.argsort(-np.array(hash.values()))[:size]
    users = np.array(hash.keys())[indx]
    weights = np.array(hash.values())[indx]
    return users, weights
    
    
#FIX THIS FUNCTION!    
def get_item_neighborhood(id_tup, ratings, userid, itemids, size, norms):
    """
    Find the nearest user's and their cosine similarity to a given user

    Arguments:
    ratings -- a ratings hash table as produced by make_ratings_hash
    userid -- the id of user neighborhood is being calculated
    size -- the number of users in userid's neighborhood
    norms -- a named vector (pandas.Series) of user l2 norms in rating space

    Returns:
    users -- a vector of the ids of the nearest users
    weights -- a vector of cosine similarities for the neighbors
    """
    items = {}
    weights = {}
    hash = {}
    for uid, iid in id_tup:
        for row in iid:
            
    for item in itemids.iteritems():
        rat = {}
        if (userid, item) in ratings: #the user has already ranked the movie
            continue
        if item not in hash:
            hash[item] = {}
        for otheritem in itemids.iteritems(): #otherwise compare 
            if otheritem == item:
                continue
            if (userid, otheritem) not in ratings:
                continue
            if otheruserid not in hash[item]:
                hash[item][otheritem] = 0
                
            hash[item][otheritem] += ratings[(userid,otheritem)] #check this line

            nx=norms[(userid, item)]
            ny=norms[(userid, otheritem)]
            hash[item][otheritem] = hash[item][otheritem]/float(nx.dot(ny))
    
        #the below needs to be per item
        indx = np.argsort(-np.array(hash[item].values()))[:size]
        items[item] = np.array(hash[item].keys())[indx]
        weights[item] = np.array(hash[item].values())[indx]
        #print 'done making item neighborhood for item'
    return items, weights
    
    '''
    
    for (otheruserid,itemid),rating in ratings.iteritems():
        if otheruserid == userid:
            continue
        if (userid, itemid) not in ratings:
            continue

        if otheruserid not in hash:
            hash[otheruserid] = 0

        hash[otheruserid] += ratings[(userid,itemid)] * rating

    for (otheruserid, val) in hash.iteritems():
        nx=norms[userid]
        ny=norms[otheruserid]
        hash[otheruserid] = hash[otheruserid]/float(nx*ny)

    indx = np.argsort(-np.array(hash.values()))[:size]
    users = np.array(hash.keys())[indx]
    weights = np.array(hash.values())[indx]
    return users, weights
    
    '''
    
    
def make_neighborhood_hash(userids, ratings, size, norms):
    neighbors = {}
    weights = {}

    for userid in userids:
        if userid not in neighbors:
            res = get_user_neighborhood(ratings, userid, size, norms)
            neighbors[userid], weights[userid] = res
    return neighbors, weights
    
def make_item_hash(ids, userids, itemids, ratings, size, norms):
    neighbors = {}
    weights = {}

    for user in userids:
        if user not in neighbors:
            res = get_item_neighborhood(ids, ratings, user, itemids, size, norms)
            neighbors[user], weights[user] = res
        print 'Done making neighborhood for user %d'%user
    print 'Done making item neighborhoods'
    return neighbors, weights

class CFilter(object):
    """A class to get ratings from collaborative filtering"""

    def __init__(self, ratings, size=20, itemsize=20):
        """
        Arguments:
        ratings -- a pandas.DataFrame of movie ratings (see prep_data)
        size -- user neighborhood size (default=20)
        """
        self.size = size
        self.ratings = make_ratings_hash(ratings)
        self.items = make_item_ratings_hash(ratings)
        print 'Done making item ratings'
        self.data = ratings

        norms = ratings[['userid','rating']].groupby('userid').aggregate(lambda x: np.sqrt(np.sum(x**2)))['rating']
        itemnorms = make_item_norms_hash(ratings)
        print 'Done making items norms hash'
        userids = ratings['userid']
        itemids = ratings['itemid']
        ids = ratings[['userid','itemid']].groupby('userid')
        self.neighbors, self.weights = make_neighborhood_hash(userids, self.ratings, size, norms)
        print 'Done making regular neighborhood hash'
        self.items, self.item_weights = make_item_hash(ids, userids, itemids, self.items, itemsize, itemnorms)
        print 'Done making items hash'

    def __repr__(self):
        return 'CFilter with %d ratings for %d users' % (len(self.ratings), len(self.neighbors))

    def get_cf_rating(self, ratings):
        """
        Get item ratings from user neighborhood

        Arguments:
        ratings -- a pandas.DataFrame of movie ratings (see prep_data)

        Returns:
        A numpy array of collaborative filter item rating's from user neighborhoods. If userid is not in
        database, 0 is returned as the item rating. Ratings are discretized from 0-5 in 0.25 increments

        """
        nratings = ratings.shape[0]
        cf_rating=np.zeros(nratings)

        for userid in self.neighbors.keys():
            indx = ratings['userid']==userid
            if np.sum(indx)==0:
                continue

            items = ratings['itemid'][indx].values
            m = len(items)
            n = len(self.neighbors[userid])

            nratings=np.zeros((m,n))
            w = np.zeros((m,n))

            for i in xrange(m):
                itemid = items[i]

                for j in xrange(n):
                    ouid = self.neighbors[userid][j]
                    if (ouid, itemid) in self.ratings:
                        nratings[i,j] = self.ratings[(ouid,itemid)]
                        w[i,j] = self.weights[userid][j]

            sw = np.sum(w,axis=1)
            keep = sw>0
            if np.sum(keep)==0:
                continue

            nratings *= w
            res = np.sum(nratings,axis=1)
            #print res, sw, keep

            res[keep.nonzero()] /= sw[keep.nonzero()]
            cf_rating[indx] = res

        return cf_rating
        
    def get_item_rating(self, ratings):
        """
        Get item ratings from item-to-item neighborhood

        Arguments:
        ratings -- a pandas.DataFrame of movie ratings (see prep_data)

        Returns:
        A numpy array of collaborative filter item rating's from user neighborhoods. If userid is not in
        database, 0 is returned as the item rating. Ratings are discretized from 0-5 in 0.25 increments

        """
        nratings = ratings.shape[0]
        cf_rating=np.zeros(nratings)

        for itemid in self.items.keys():
            indx = ratings['itemid']==itemid
            if np.sum(indx)==0:
                continue

            items = ratings['itemid'][indx].values
            m = len(items)
            n = len(self.items[itemid])

            nratings=np.zeros((m,n))
            w = np.zeros((m,n))

            for i in xrange(m):
                itemid = items[i]

                for j in xrange(n):
                    ouid = self.items[itemid][j]
                    if (ouid, itemid) in self.ratings:
                        nratings[i,j] = self.ratings[(ouid,itemid)]
                        w[i,j] = self.weights[itemid][j]

            sw = np.sum(w,axis=1)
            keep = sw>0
            if np.sum(keep)==0:
                continue

            nratings *= w
            res = np.sum(nratings,axis=1)
            #print res, sw, keep

            res[keep.nonzero()] /= sw[keep.nonzero()]
            cf_rating[indx] = res

        return cf_rating

