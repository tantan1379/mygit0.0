class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def InitLinkList(arr):
    if(len(arr) == 0):
        print("invalid input!")
        return None

    head = ListNode(arr[0])
    cur = head
    for a in arr[1:]:
        cur.next = ListNode(a)
        cur = cur.next
    return head


def ForeachLinkList(head):
    cur = head
    while(cur):
        if(cur.next == None):
            print(cur.val)
        else:
            print("{}->".format(cur.val), end="")
        cur = cur.next


if __name__ == "__main__":
    ll = InitLinkList([1,2,3])
    ForeachLinkList(ll)
