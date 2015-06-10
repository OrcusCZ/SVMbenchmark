#include <stdio.h>
#include <stdlib.h>
#include "../../include/Cache.h"



Cache::DirectoryEntry::DirectoryEntry()
{
  status = NEVER;
}


/**
 * Constructor of the Cache
 */
Cache::Cache(int nPointsIn, int cacheSizeIn)
{
  directory.reserve(nPointsIn);
  nPoints = nPointsIn;
  cacheSize = cacheSizeIn;
  occupancy = 0;
  hits = 0;
  compulsoryMisses = 0;
  capacityMisses = 0;
  for(size_t i=0; i < nPoints; i++) directory.push_back(DirectoryEntry()); 
}

/**
 * Destructor of the Cache
 */
Cache::~Cache()
{

}

/**
 * Search for an index in the cache
 */
void Cache::search(const int index, int &offset, bool &compute)
{
  DirectoryEntry currentEntry = directory[index];
  if (currentEntry.status == DirectoryEntry::INCACHE)
  {
    offset = currentEntry.location;
    compute = false;
    return;
  }
  compute = true;
  return;
}

/**
 * Find data in the cache
 */
void Cache::findData(const int index, int &offset, bool &compute)
{
  std::vector<DirectoryEntry>::iterator iCurrentEntry = directory.begin() + index;
  if (iCurrentEntry->status == DirectoryEntry::INCACHE)
  {
    hits++;
    if (iCurrentEntry->lruListEntry == lruList.begin())
    {
      offset = iCurrentEntry->location;
      compute = false;
      return;
    }
    lruList.erase(iCurrentEntry->lruListEntry);
    lruList.push_front(index);
    iCurrentEntry->lruListEntry = lruList.begin();
    offset = iCurrentEntry->location;
    compute = false;
    return;
  }

  //Cache Miss
  if (occupancy < cacheSize)
  {
    //Cache has empty space
    compulsoryMisses++;
    iCurrentEntry->location = occupancy;
    iCurrentEntry->status = DirectoryEntry::INCACHE;
    lruList.push_front(index);
    iCurrentEntry->lruListEntry = lruList.begin();
    occupancy++;
    offset = iCurrentEntry->location;
    compute = true;
    return;
  }

  //Cache is full
  if (iCurrentEntry->status == DirectoryEntry::NEVER)
  {
    compulsoryMisses++;
  }
  else
  {
    capacityMisses++;
  }

  int expiredPoint = lruList.back();
  lruList.pop_back();

  directory[expiredPoint].status = DirectoryEntry::EVICTED;
  int expiredLine = directory[expiredPoint].location;
  iCurrentEntry->status = DirectoryEntry::INCACHE;
  iCurrentEntry->location = expiredLine;
  lruList.push_front(index);
  iCurrentEntry->lruListEntry = lruList.begin();

  offset = iCurrentEntry->location;
  compute = true;
  return;
}

/**
 * Print statistics of the cache
 */
void Cache::printStatistics()
{
	int accesses = hits + compulsoryMisses + capacityMisses;
	float rate= (float)hits/(float) accesses;
	printf("%d accesses, %d hits, %d compulsory misses, %d capacity misses, hit rate %f\n", accesses, hits, compulsoryMisses, capacityMisses, rate);
	return;
}

/**
 * Print contents of the cache
 */
void Cache::printCache()
{
  int accesses = hits + compulsoryMisses + capacityMisses;
  float hitPercent = (float)hits*100.0/float(accesses);
  float compulsoryPercent = (float)compulsoryMisses*100.0/float(accesses);
  float capacityPercent = (float)capacityMisses*100.0/float(accesses);

  printf("Cache hits: %f compulsory misses: %f capacity misses %f\n", hitPercent, compulsoryPercent, capacityPercent);
  for(int i = 0; i < nPoints; i++)
  {
    if (directory[i].status == DirectoryEntry::INCACHE)
    {
      printf("Row %d: present @ cache line %d\n", i, directory[i].location);
    } else {
      printf("Row %d: not present\n", i);
    }
  }
  printf("----\n");
  std::list<int>::iterator i = lruList.begin();
  for(;i != lruList.end(); i++) {
    printf("Offset: %d\n", *i);
  }
}

