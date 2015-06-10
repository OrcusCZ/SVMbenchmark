#include <stdio.h>
#include <stdlib.h>
#include "Cache.h"

CacheGpuSVM::DirectoryEntry::DirectoryEntry() {
  status = NEVER;
}

CacheGpuSVM::CacheGpuSVM(int nPointsIn, int cacheSizeIn) {
  directory.reserve(nPointsIn);
  nPoints = nPointsIn;
  cacheSize = cacheSizeIn;
  occupancy = 0;
  hits = 0;
  compulsoryMisses = 0;
  capacityMisses = 0;
}

CacheGpuSVM::~CacheGpuSVM() {
 
}

void CacheGpuSVM::search(const int index, int &offset, bool &compute) {
  DirectoryEntry currentEntry = directory[index];
  if (currentEntry.status == DirectoryEntry::INCACHE) {
    offset = currentEntry.location;
    compute = false;
    return;
  }
  compute = true;
  return;
}

void CacheGpuSVM::findData(const int index, int &offset, bool &compute) {
  size_t i;
  bool pushed = false;

  DirectoryEntry de = DirectoryEntry();
  DirectoryEntry *iCurrentEntry;

  de.status = 1;

  if (directory.size() > index) {
    iCurrentEntry = &(directory[index]);
    if (iCurrentEntry->status == DirectoryEntry::INCACHE) {
      hits++;
      if (iCurrentEntry->lruListEntry == lruList.begin()) {
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
  }

  //CacheGpuSVM Miss
  if (occupancy < cacheSize) {
    for (i = directory.size(); i <= index; i++) {
      if (pushed == false) {
        pushed = true;
      }
	  directory.push_back(DirectoryEntry());
	}
	if (pushed == true) {
      iCurrentEntry = &(directory[index]);
    }
    //CacheGpuSVM has empty space
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
 
  //CacheGpuSVM is full
  if (iCurrentEntry->status == DirectoryEntry::NEVER) {
    compulsoryMisses++;
  } else {
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

void CacheGpuSVM::printStatistics() {
	int accesses = hits + compulsoryMisses + capacityMisses;
	printf("%d accesses, %d hits, %d compulsory misses, %d capacity misses\n", accesses, hits, compulsoryMisses, capacityMisses);
	return;
}

void CacheGpuSVM::printCache() {
  int accesses = hits + compulsoryMisses + capacityMisses;
  float hitPercent = (float)hits*100.0/float(accesses);
  float compulsoryPercent = (float)compulsoryMisses*100.0/float(accesses);
  float capacityPercent = (float)capacityMisses*100.0/float(accesses);
  
  printf("CacheGpuSVM hits: %f compulsory misses: %f capacity misses %f\n", hitPercent, compulsoryPercent, capacityPercent);
  for(int i = 0; i < nPoints; i++) {
    if (directory[i].status == DirectoryEntry::INCACHE) {
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

// int main(int argc, char** argv) {
//   int nPoints = 10;
//   int cacheSize = 5;
//   CacheGpuSVM theCache(nPoints, cacheSize);
//   for(int i = 0; i < nPoints; i++) {
//     int offset;
//     bool rebuild;
//     theCache.findData(i, offset, rebuild);
//     if (rebuild == false) { 
//       printf("point %d found in cache at row %d\n", i, offset);
//     } else {
//       printf("point %d not in cache.  Row %d allocated\n", i, offset);
//     }
//     //theCache.printCache();

//   }

//   for (int i = 0; i < 5; i++) {
//     int offset;
//     bool rebuild;
//     int j = rand() % nPoints;
//     theCache.findData(j, offset, rebuild);
//     if (rebuild == false) { 
//       printf("point %d found in cache at row %d\n", j, offset);
//     } else {
//       printf("point %d not in cache.  Row %d allocated\n", j, offset);
//     }
//   }
//   printf("-------\n");
//   theCache.printCache();
  
// }
