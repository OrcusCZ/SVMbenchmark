#ifndef CACHEH
#define CACHEH

#include <vector>
#include <list>

class Cache
{
 public:
  Cache(int nPointsIn, int cacheSizeIn);
  ~Cache();
  void findData(const int index, int &offset, bool &compute);
	void search(const int index, int &offset, bool &compute);
  void printCache();
	void printStatistics();
private:
  int nPoints;
  int cacheSize;
  class DirectoryEntry
  {
  public:
    enum {NEVER, EVICTED, INCACHE};
    DirectoryEntry();
    int status;
    int location;
    std::list<int>::iterator lruListEntry;
  };

  std::vector<DirectoryEntry> directory;
  std::list<int> lruList;
  int occupancy;
  int hits;
  int compulsoryMisses;
  int capacityMisses;
};

#endif
