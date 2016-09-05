import itertools

class max_binary_heap(object):
    
    def __init__(self):
        self.heap_list = [0]
        self.heap_size = 0
        self.counter = itertools.count()
        self.d = {}
    
    def is_empty(self):
        if self.heap_size == 0:
            return True
    
    def swap(self,i,j):
        self.d[self.heap_list[i][2]] = j
        self.d[self.heap_list[j][2]] = i
        self.heap_list[i],self.heap_list[j] = self.heap_list[j],self.heap_list[i]
        
    
    def size(self):
        return self.heap_size
    
    def get_max(self):
        return self.heap_list[1]
    
    def perc_up(self,i):
        while i/2 > 0:
            if self.heap_list[i] > self.heap_list[i/2]:
                self.swap(i,i/2)
            i = i/2
            
    def perc_down(self,i):
        while i*2 <= self.heap_size:
            mc = self.max_child(i)
            if self.heap_list[i] < self.heap_list[mc]:
                self.swap(i,mc)
            i = mc
    
    def max_child(self,i):
        if i*2 + 1 > self.heap_size:
            return 2*i
        elif self.heap_list[2*i] > self.heap_list[2*i+1]:
            return 2*i
        else:
            return 2*i +1
        
    def insert(self, priority, task):
        if task not in self.d:
             self.d[task] = self.heap_size + 1
             count = next(self.counter)
             entry = [priority, count, task]
             self.heap_list.append(entry)
             self.heap_size = self.heap_size + 1
             self.perc_up(self.heap_size)
        else:
            index = self.d[task]
            cmp = priority - self.heap_list[index][0]
            count = next(self.counter)
            entry = [priority, count, task]
            self.heap_list[index] = entry
            if cmp < 0.0:
                self.perc_down(index)
            else:
                self.perc_up(index)
             
    def del_max(self):
        if self.is_empty() is True:
            return "Heap is Empty"
        retval = self.heap_list[1]
        self.heap_list[1] = self.heap_list[self.heap_size]
        self.d[self.heap_list[1][2]] = 1
        self.heap_size = self.heap_size - 1
        self.heap_list.pop()
        self.perc_down(1)
        del self.d[retval[2]]
        return retval

