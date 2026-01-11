#ifndef MYQUEUE
#define MYQUEUE

#include <bits/stdc++.h>
using namespace std;

#define MaxSize 100000000

template <typename T>
class Myqueue
{
public:
    Myqueue() : front(-1), rear(-1), count(0)
    {
        data = new T[MaxSize];
    }

    void qin(T value)
    {
        rear = (rear + 1) % MaxSize;
        data[rear] = value;
        count++;
    }

    T qout()
    {
        front = (front + 1) % MaxSize;
        count--;
        return data[front];
    }

    bool empty() const
    {
        return front == rear;
    }

    bool full() const
    {
        return (rear + 1) % MaxSize == front;
    }

    ~Myqueue()
    {
        delete[] data;
    }

    T *data;
    int front, rear;
    size_t count;
};

#endif
