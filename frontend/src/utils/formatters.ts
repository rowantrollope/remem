export const formatTimestamp = (timestamp: string): string => {
    return new Date(timestamp).toLocaleString()
}
