import itertools

class max_binary_heap(object):
    
    def __init__(self):
        self.heap_list = [0]
        self.heap_size = 0
        self.d = {}
    
    def swap(self,i,j):
        self.heap_list[i], self.heap_list[j] = self.heap_list[j], self.heap_list[i]
        self.heap_list[i][2] = i
        self.heap_list[j][2] = j
    
    def is_empty(self):
        if self.heap_size == 0:
            return True
    
    def size(self):
        return self.heap_size
    
    def get_max(self):
        return self.heap_list[1]
    
    def perc_up(self,i):
        while i/2 > 0:
            if self.heap_list[i] > self.heap_list[i/2]:
                self.heap_list[i],self.heap_list[i/2] = self.heap_list[i/2],self.heap_list[i]
            i = i/2
            
    def perc_down(self,i):
        while i*2 <= self.heap_size:
            mc = self.max_child(i)
            if self.heap_list[i] < self.heap_list[mc]:
                self.heap_list[i],self.heap_list[mc] = self.heap_list[mc],self.heap_list[i]
            i = mc
    
    def max_child(self,i):
        if i*2 + 1 > self.heap_size:
            return 2*i
        elif self.heap_list[2*i] > self.heap_list[2*i+1]:
            return 2*i
        else:
            return 2*i +1
        
    def insert(self,key,value):
        self.heap_list.append(item)
        self.heap_size = self.heap_size + 1
        self.perc_up(self.heap_size)
        
        if key in self.d:
            self.pop(key)
        wrapper = [value, key, len(self)]
        self.d[key] = wrapper
        self.heap.append(wrapper)
        self._decrease_key(len(self.heap)-1)
        
        
        
        
        
        
    
    def del_max(self):
        if self.is_empty() is True:
            return "Heap is Empty"
        retval = self.heap_list[1]
        self.heap_list[1] = self.heap_list[self.heap_size]
        self.heap_size = self.heap_size - 1
        self.heap_list.pop()
        self.perc_down(1)
        return retval
    
    def build_heap(self,alist):
        i = len(alist) / 2
        self.heap_size = len(alist)
        self.heap_list = [0] + alist[:]
        while (i > 0):
            self.perc_down(i)
            i = i - 1

class priority_queue(object):
    
    def __init__(self):
        self.pq = max_binary_heap()
        self.counter = itertools.count()
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
    
    def add_task(self,task, priority=0):
        'Add a new delta or update the priority of an existing delta'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        self.pq.insert(entry)
        
    def remove_task(self,task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED
    
    def del_max_task(self):
        'Remove and return the highest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = self.pq.del_max()
            if task is not REMOVED:
                del entry_finder[task]
            return task
        raise KeyError('pop from an empty priority queue')
        
    def build_queue(self,tasks,priorities):
        "Build a priority queue from tasks and priorities"
        assert len(tasks) == len(priorities)
        counts = [0]*len(tasks)
        initial_q = [[x,y,z] for x,y,z in zip(priorities,counts,tasks)]
        self.pq.build_heap(initial_q)
        self.entry_finder = {key:value for key,value in zip(tasks,initial_q) }
    
