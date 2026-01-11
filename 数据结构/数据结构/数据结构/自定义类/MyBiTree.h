#ifndef MYBITREE
#define MYBITREE
#include <bits/stdc++.h>
#include "Myqueue.h"

using namespace std;

template <typename T>
struct BiNode
{
    T data;
    BiNode *lchild, *rchild;
    BiNode(T val)
    {
        this->data = val;
        this->lchild = nullptr;
        this->rchild = nullptr;
    }
    BiNode()
    {
        this->lchild = nullptr;
        this->rchild = nullptr;
    }
};

template <typename T>
class LBiTree
{
public:
    BiNode<T> *root;
    LBiTree(const LBiTree<T> *other)
    {
        this->root = other->root;
    }
    LBiTree()
    {
        BiNode<T> *node = new BiNode<T>();
        this->root = node;
    }

    LBiTree(T value)
    {
        BiNode<T> *node = new BiNode<T>(value);
        this->root = node;
    }
    LBiTree(const vector<T> &preorder, const vector<T> &inorder)
    {
        root = BuildHelper(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
    }

    BiNode<T> *BuildHelper(const vector<T> &preorder, int pre_start, int pre_end, const vector<T> &inorder, int in_start, int in_end)
    {
        if (pre_start > pre_end || in_start > in_end)
        {
            return nullptr;
        }

        T root_value = preorder[pre_start];
        BiNode<T> *root = new BiNode<T>(root_value);

        int root_index = in_start;
        while (root_index <= in_end && inorder[root_index] != root_value)
        {
            root_index++;
        }

        int leftTreeSize = root_index - in_start;

        root->lchild = BuildHelper(preorder, pre_start + 1, pre_start + leftTreeSize, inorder, in_start, root_index - 1);
        root->rchild = BuildHelper(preorder, pre_start + leftTreeSize + 1, pre_end, inorder, root_index + 1, in_end);

        return root;
    }

    LBiTree<T> &operator=(const LBiTree<T> &tree)
    {
        if (this == &tree)
        {
            return *this;
        }
        clear(root);
        root = deepCopyHelper(tree.root);
        return *this;
    }
    BiNode<T> *deepCopyHelper(BiNode<T> *node)
    {
        if (node == nullptr)
        {
            return nullptr;
        }

        BiNode<T> *newNode = new BiNode<T>(node->data);

        newNode->lchild = deepCopyHelper(node->lchild);
        newNode->rchild = deepCopyHelper(node->rchild);

        return newNode;
    }
    bool is_empty()
    {
        return (root == nullptr);
    }
    void clearr()
    {
        clear(this->root);
    }
    ~LBiTree()
    {
        clear(this->root);
    }
};

template <typename T>
void pre_order1(BiNode<T> *node)
{
    stack<BiNode<T> *> lst;
    BiNode<T> *p = node;
    while (p != nullptr || !lst.empty())
    {
        while (p != nullptr)
        {
            lst.push(p);
            cout << p->data << " ";
            p = p->lchild;
        }
        if (!lst.empty())
        {
            p = lst.top();
            lst.pop();
            p = p->rchild;
        }
    }
}

template <typename T>
void pre_order(BiNode<T> *node)
{
    if (node == nullptr)
    {
        return;
    }
    cout << node->data << " ";
    pre_order(node->lchild);
    pre_order(node->rchild);
}

template <typename T>
void in_order1(BiNode<T> *node)
{
    stack<BiNode<T> *> lst;
    BiNode<T> *p = node;
    while (p != nullptr || !lst.empty())
    {
        while (p != nullptr)
        {
            lst.push(p);
            p = p->lchild;
        }
        if (!lst.empty())
        {
            p = lst.top();
            lst.pop();
            cout << p->data << " ";
            p = p->rchild;
        }
    }
}

template <typename T>
void in_order(BiNode<T> *node)
{
    if (node == nullptr)
    {
        return;
    }
    pre_order(node->lchild);
    cout << node->data << " ";
    pre_order(node->rchild);
}

template <typename T>
void post_order1(BiNode<T> *node)
{
    stack<BiNode<T> *> lst;
    BiNode<T> *p = node;
    while (p != nullptr || !lst.empty())
    {
        while (p != nullptr)
        {
            lst.push(p);
            p = p->lchild;
        }
        BiNode<T> *topp = lst.top();
        BiNode<T> *pre_top = nullptr;

        while (!lst.empty() && topp->rchild == pre_top)
        {
            cout << topp->data << " ";
            pre_top = topp;
            lst.pop();
            if (!lst.empty())
            {
                topp = lst.top();
            }
            else
            {
                topp = nullptr;
            }
        }

        if (topp != nullptr)
        {
            p = lst.top()->rchild;
        }
    }
}

template <typename T>
void post_order(BiNode<T> *node)
{
    if (node == nullptr)
    {
        return;
    }
    pre_order(node->lchild);
    pre_order(node->rchild);
    cout << node->data << " ";
}

template <typename T>
void level_order(LBiTree<T> *tree)
{
    if (tree->root == nullptr)
    {
        cout << "None" << endl;
        return;
    }
    Myqueue<BiNode<T> *> *lst = new Myqueue<BiNode<T> *>;
    BiNode<T> *node = tree->root;
    lst->qin(node);
    while (!lst->empty())
    {
        BiNode<T> *temp = lst->qout();
        cout << temp->data << " ";
        if (temp->lchild)
        {
            lst->qin(temp->lchild);
        }
        if (temp->rchild)
        {
            lst->qin(temp->rchild);
        }
    }
    cout << endl;
}

template <typename T>
void node_num(LBiTree<T> *tree)
{
    Myqueue<BiNode<T> *> *lst = new Myqueue<BiNode<T> *>;
    BiNode<T> *node = tree->root;
    lst->qin(node);
    int node_cnt = 0;
    int leaf_cnt = 0;
    while (!lst->empty())
    {
        BiNode<T> *temp = lst->qout();
        node_cnt++;
        if (temp->lchild)
        {
            lst->qin(temp->lchild);
        }
        if (temp->rchild)
        {
            lst->qin(temp->rchild);
        }
        if ((!temp->rchild) && (!temp->lchild))
        {
            leaf_cnt++;
        }
    }
    cout << "The node number is: " << node_cnt << endl;
    cout << "The leaf node number is: " << leaf_cnt << endl;
}

template <typename T>
void clear(BiNode<T> *&node)
{
    if (node == nullptr)
    {
        return;
    }
    clear(node->lchild);
    clear(node->rchild);
    delete node;
    node = nullptr;
}

template <typename T>
int width(LBiTree<T> *tree)
{
    if (tree->root == nullptr)
    {
        return 0;
    }
    BiNode<T> *node = tree->root;
    Myqueue<BiNode<T> *> *que = new Myqueue<BiNode<T> *>();
    que->qin(node);
    int maxwidth = 0;
    while (!que->empty())
    {
        int ssize = que->count;
        maxwidth = max(maxwidth, ssize);
        for (int i = 0; i < ssize; i++)
        {
            BiNode<T> *temp = que->qout();
            if (temp->lchild)
            {
                que->qin(temp->lchild);
            }
            if (temp->rchild)
            {
                que->qin(temp->rchild);
            }
        }
    }
    return maxwidth;
}

template <typename T>
int height(BiNode<T> *node)
{
    int max_height = 0;
    if (node == nullptr)
    {
        return 0;
    }
    int leftheight = height(node->lchild);
    int rightheight = height(node->rchild);
    return max(leftheight, rightheight) + 1;
}

template <typename T>
BiNode<T> *insert(BiNode<T> *root, T value)
{
    if (root == nullptr)
    {
        return new BiNode<T>(value);
    }

    int leftheight = height(root->lchild);
    int rightheight = height(root->rchild);
    if (leftheight < rightheight)
    {
        root->lchild = insert(root->lchild, value);
    }
    else
    {
        root->rchild = insert(root->rchild, value);
    }
    return root;
}

template <typename T>
bool mirror_image(BiNode<T> *temp1, BiNode<T> *temp2)
{

    if (temp1 == nullptr && temp2 == nullptr)
    {
        return true;
    }
    if (temp1 == nullptr || temp2 == nullptr)
    {
        return false;
    }
    if (temp1->data != temp2->data)
    {
        return false;
    }

    return mirror_image(temp1->lchild, temp2->rchild) && mirror_image(temp1->rchild, temp2->lchild);
}
#endif